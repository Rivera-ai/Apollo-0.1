from VideoAutoencoder import AdaptiveEfficientVideoAutoencoder, save_video_tensor
from PrometheusCore import Prometheus
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torch
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchmetrics.functional as metrics
import torch.nn as nn
from einops import rearrange

def collate_fn(batch):
    """Solo toma los videos del batch, ignorando el texto"""
    videos = [item[1] for item in batch]  # item[1] es el video tensor
    return torch.stack(videos)

class VideoProcessor:
    def __init__(self, target_size=(240, 426), fps=15, duration=2):
        self.target_size = target_size
        self.fps = fps
        self.duration = duration
        self.num_frames = fps * duration
        self.transform = self.normalize_video

    def normalize_video(self, video):
        """
        Normaliza y asegura las dimensiones correctas del video
        Input/Output: tensor de forma [C, T, H, W]
        """
        # Print shape for debugging
        #print(f"Input video shape: {video.shape}")

        # Ensure correct temporal dimension
        if video.shape[1] != self.num_frames:
            # Interpolate temporal dimension
            video = F.interpolate(
                video.unsqueeze(0),
                size=(self.num_frames, video.shape[2], video.shape[3]),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        # Ensure correct spatial dimensions
        if video.shape[2:] != self.target_size:
            # Interpolate spatial dimensions
            video = F.interpolate(
                video.unsqueeze(0),
                size=(video.shape[1], *self.target_size),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        # Normalize to [-1, 1]
        video = (video * 2) - 1

        # Print final shape for verification
        #print(f"Output video shape: {video.shape}")

        return video

    def process_batch(self, batch):
        """
        Process a batch of videos to ensure consistent dimensions
        """
        if isinstance(batch, torch.Tensor):
            # If batch is already stacked
            if batch.shape[-2:] != self.target_size or batch.shape[2] != self.num_frames:
                batch = F.interpolate(
                    batch,
                    size=(self.num_frames, *self.target_size),
                    mode='trilinear',
                    align_corners=False
                )
            return batch
        else:
            # If batch is a list
            processed = [self.normalize_video(video) for video in batch]
            return torch.stack(processed)

class VideoDataset(Dataset):
    def __init__(self, csv_path, video_folder, processor):
        self.video_folder = Path(video_folder)
        self.processor = processor
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 1024
        # Verificar existencia de archivos
        self.data = self.data[self.data.apply(
            lambda x: (self.video_folder / f"{x['video_name']}.mp4").exists(),
            axis=1
        )]
        print(f"Dataset cargado con {len(self.data)} videos válidos")

    def __len__(self):
        return len(self.data)

    def load_video(self, video_path):
        try:
            import av
            with av.open(str(video_path)) as container:  # Usar context manager
                stream = container.streams.video[0]

            # Contar frames primero sin guardarlos
                total_frames = stream.frames
                if total_frames == 0:
                # Si no podemos obtener el conteo directamente, contamos manualmente
                    total_frames = sum(1 for _ in container.decode(video=0))
                    container.seek(0)  # Regresar al inicio


                if total_frames == 0:
                    raise ValueError("No frames found in video")

            # Calcular índices para muestreo uniforme
                if total_frames >= self.processor.num_frames:
                    indices = np.linspace(0, total_frames-1, self.processor.num_frames, dtype=int)
                else:
                    indices = np.arange(total_frames)

            # Crear tensor de salida
                video_tensor = torch.zeros(3, self.processor.num_frames, *self.processor.target_size)

            # Procesar frames uno por uno
                current_frame = 0
                for i, frame in enumerate(container.decode(video=0)):
                    if i not in indices:
                        continue

                # Procesar solo los frames que necesitamos
                    output_idx = np.where(indices == i)[0][0]
                    if output_idx >= self.processor.num_frames:
                        break

                # Convertir frame a tensor directamente
                    img = frame.to_ndarray(format='rgb24')
                    img = torch.from_numpy(img).float() / 255.0

                # Redimensionar si es necesario
                    if img.shape[0:2] != self.processor.target_size:
                        img = F.interpolate(
                            img.permute(2, 0, 1).unsqueeze(0),
                            size=self.processor.target_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    else:
                        img = img.permute(2, 0, 1)

                    video_tensor[:, output_idx] = img
                    current_frame = output_idx

            # Si faltan frames, repetir el último
                if current_frame + 1 < self.processor.num_frames:
                    video_tensor[:, current_frame+1:] = video_tensor[:, current_frame].unsqueeze(1).expand(
                        -1, self.processor.num_frames - (current_frame + 1), -1, -1
                    )

                return self.processor.transform(video_tensor)

        except Exception as e:
            print(f"Error cargando video {video_path}: {str(e)}")
            return torch.zeros(3, self.processor.num_frames, *self.processor.target_size)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = self.video_folder / f"{row['video_name']}.mp4"

        video_tensor = self.load_video(video_path)

        # Procesar texto
        text = row['answer']
        text_encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_tensor = text_encoded.squeeze(0)

        return text_tensor, video_tensor

class VideoEncoderWrapper(nn.Module):
    def __init__(self, encoder_blocks, attention):
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.attention = attention
        
    def forward(self, x):
        # Procesar a través de los bloques del encoder
        h = x
        for block in self.encoder_blocks:
            h = block(h)
            
        # Aplicar attention
        att = self.attention(h)
        h = h * att
        
        # Reorganizar de [B, C, T, H, W] a [B, T, H, W, C]
        h = rearrange(h, 'b c t h w -> b t h w c')
        
        return h

class VideoDecoderWrapper(nn.Module):
    def __init__(self, decoder_blocks, final_adjust=None):
        super().__init__()
        self.decoder_blocks = decoder_blocks
        self.final_adjust = final_adjust
        
    def forward(self, x):
        # Reorganizar de [B, T, H, W, C] a [B, C, T, H, W]
        h = rearrange(x, 'b t h w c -> b c t h w')
        
        for block in self.decoder_blocks:
            h = block(h)
            
        if self.final_adjust is not None:
            h = self.final_adjust(h)
            
        return h

def TrainAutoEncoderAdapt240p():
    """
    Entrena el AutoEncoder adaptativo específicamente para videos de 240p y 10 segundos
    """
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_epochs = 1
    save_every = 50

    # Configuración específica para 240p
    resolution = (240, 426)
    fps = 15
    duration = 10

    # Setup directorios
    results_folder = Path('./results_adaptive_240p')
    results_folder.mkdir(exist_ok=True, parents=True)

    # Inicializar procesador
    processor = VideoProcessor(
        target_size=resolution,
        fps=fps,
        duration=duration
    )

    # Dataset
    dataset = VideoDataset(
        csv_path="/content/Videos/datos_videos.csv",
        video_folder="/content/Videos/Test_Videos/",
        processor=processor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Modelo y optimizador - Asegurando que todo esté en float32 inicialmente
    model = AdaptiveEfficientVideoAutoencoder(dim_latent=128, duration=10, quality='240p').to(device)
    model.print_model_info()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)
    scaler = GradScaler()

    # Variables de tracking
    best_loss = float('inf')
    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} | 240p Training')

        for videos in pbar:
            try:
                # Procesar videos y moverlos a GPU en float32
                videos = processor.process_batch(videos)
                videos = videos.to(device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)

                # Usar autocast para precisión mixta
                with autocast():
                    reconstructed = model(videos)
                    assert reconstructed.shape == videos.shape, \
                        f"Shape mismatch: reconstructed {reconstructed.shape} vs input {videos.shape}"

                    # Pérdida combinada
                    recon_loss = F.l1_loss(reconstructed, videos)
                    ssim_loss = 1 - metrics.structural_similarity_index_measure(
                        reconstructed,
                        videos,
                        data_range=2.0
                    )
                    loss = 0.7 * recon_loss + 0.3 * ssim_loss

                # Backward y optimize con scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Logging
                pbar.set_description(
                    f'Epoch {epoch} | 240p | Loss: {loss.item():.4f} | '
                    f'Recon: {recon_loss.item():.4f} | SSIM: {ssim_loss.item():.4f}'
                )

                # Guardar checkpoints y muestras
                if global_step % save_every == 0:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, results_folder / 'best_model_240p.pt')

                    with torch.no_grad():
                        save_video_tensor(
                            reconstructed[0],
                            results_folder / f'recon_video_240p_{global_step}.mp4',
                            fps=processor.fps
                        )

                global_step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("\nOOM Error, skipping batch")
                    continue
                raise e

def TrainApolloVideo():
    """ Entrena el modelo Apollo para generar videos """
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_epochs = 100
    save_every = 10

    # Configuración específica para 240p
    resolution = (240, 426)
    fps = 15
    duration = 10

    # Setup directorios
    results_folder = Path('./results_prometheus')
    results_folder.mkdir(exist_ok=True, parents=True)

    # Inicializar procesador
    processor = VideoProcessor(
        target_size=resolution,
        fps=fps,
        duration=duration
    )

    # Dataset
    dataset = VideoDataset(
        csv_path="/teamspace/studios/this_studio/datos_videos.csv",
        video_folder="/teamspace/studios/this_studio/VideoDetailCaption/Test_Videos/",
        processor=processor
    )

    # Crear dataloader sin collate_fn personalizado para mantener texto y video
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    # Cargar AutoEncoder pre-entrenado
    video_autoencoder = AdaptiveEfficientVideoAutoencoder(
        dim_latent=128,
        duration=duration,
        quality='240p'
    ).to(device)

    # Cargar checkpoint
    autoencoder_checkpoint = torch.load('./results_adaptive_240p/best_model_240p.pt')
    video_autoencoder.load_state_dict(autoencoder_checkpoint['model_state_dict'])
    print("AutoEncoder cargado exitosamente")

    encoder = VideoEncoderWrapper(
        video_autoencoder.encoder_blocks,
        video_autoencoder.attention
    )
    
    decoder = VideoDecoderWrapper(
        video_autoencoder.decoder_blocks,
        video_autoencoder.final_adjust
    )

    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    # Inicializar Prometheus
    model = Prometheus(
        num_text_tokens=50257,  # Tamaño del vocabulario de GPT2
        transformer=dict(
            dim=768,
            depth=12,
            heads=12,
            dim_head=64
        ),
        modality_encoder=encoder,
        modality_decoder=decoder,
        dim_latent=128,
        modality_num_dim=3,  # Para video (tiempo, altura, ancho)
        modality_default_shape=(150, 240, 426),  # Para 240p @ 15fps por 10s
        channel_first_latent=False,
        add_pos_emb=True
    ).to(device)

    # Optimizador y scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)
    scaler = GradScaler()

    # Variables de tracking
    best_loss = float('inf')
    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} | Prometheus Training')

        for batch in pbar:
            try:
                text_tensors, video_tensors = batch

                # Mover a GPU
                text_tensors = text_tensors.to(device)
                video_tensors = video_tensors.to(device)

                # Preparar formato de entrada para Prometheus
                # Cada muestra debe ser [text_tensor, (0, video_tensor)]
                batch_samples = []
                for text, video in zip(text_tensors, video_tensors):
                    batch_samples.append([
                        text,
                        (0, video)  # 0 indica el primer tipo de modalidad
                    ])

                optimizer.zero_grad(set_to_none=True)

                # Forward pass con precisión mixta
                with autocast():
                    loss = model(
                        batch_samples,
                        return_loss=True
                    )

                # Backward y optimize con scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Logging
                pbar.set_description(
                    f'Epoch {epoch} | Loss: {loss.item():.4f}'
                )

                # Guardar checkpoints
                if global_step % save_every == 0:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, results_folder / 'best_model.pt')

                    # Checkpoint regular
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, results_folder / f'checkpoint_{global_step}.pt')

                global_step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("\nOOM Error, skipping batch")
                    continue
                raise e

if __name__ == '__main__':
    #print("Training AutoEncoder 240p 10s")
    #TrainAutoEncoderAdapt240p()
    print("Training Prometheus with Video")
    TrainApolloVideo()