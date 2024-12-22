from Models.VideoAutoEncoder import EfficientVideoAutoencoder, save_video_tensor
from Models.Model.PrometheusModel import Prometheus
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torchmetrics.functional as metrics
from itertools import cycle
from datasets import load_dataset

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
        print(f"Input video shape: {video.shape}")
        
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
        print(f"Output video shape: {video.shape}")
        
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
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            
            # Obtener todos los frames primero
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame)
            
            total_frames = len(frames)
            if total_frames == 0:
                raise ValueError("No frames found in video")
                
            # Calcular índices para muestreo uniforme
            if total_frames >= self.processor.num_frames:
                indices = np.linspace(0, total_frames-1, self.processor.num_frames, dtype=int)
            else:
                indices = np.arange(total_frames)
                
            # Crear tensor de salida
            video_tensor = torch.zeros(3, self.processor.num_frames, *self.processor.target_size)
            
            # Procesar frames seleccionados
            for i, idx in enumerate(indices):
                if i >= self.processor.num_frames:
                    break
                    
                if idx >= len(frames):
                    # Usar último frame si nos pasamos
                    img = frames[-1].to_ndarray(format='rgb24')
                else:
                    img = frames[idx].to_ndarray(format='rgb24')
                
                # Convertir a tensor y normalizar
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
                
                video_tensor[:, i] = img
            
            # Si faltan frames, repetir el último
            if len(indices) < self.processor.num_frames:
                video_tensor[:, len(indices):] = video_tensor[:, -1].unsqueeze(1).expand(-1, self.processor.num_frames - len(indices), -1, -1)
            
            # Liberar memoria
            del frames
            container.close()
            
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
        text_encoded = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
        text_tensor = torch.tensor(text_encoded, dtype=torch.long)
        
        return text_tensor, video_tensor

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.data = []
        for item in data:
            text = item['text']
            if not isinstance(text, str):
                text = str(text)
            bytes_array = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
            self.data.extend(bytes_array)
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.data_length = len(self.data)

    def __len__(self):
        return self.data_length // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data_length - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

def TrainAutoEncoder():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_epochs = 100
    save_every = 500  # Guardar cada N pasos
    
    # Setup directorios
    results_folder = Path('./results')
    results_folder.mkdir(exist_ok=True, parents=True)
    
    # Dataset y modelo
    processor = VideoProcessor(target_size=(240, 426))  # Ajustado para 240p
    dataset = VideoDataset(
        csv_path="/teamspace/studios/this_studio/datos_videos.csv",
        video_folder="/teamspace/studios/this_studio/VideoDetailCaption/Test_Videos/",
        processor=processor
    )
    
    # Modificar el __getitem__ para solo retornar el video
    original_getitem = dataset.__getitem__
    dataset.__getitem__ = lambda idx: original_getitem(idx)[1]
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    # Modelo y optimizador
    model = EfficientVideoAutoencoder(dim_latent=128).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)
    scaler = GradScaler()
    
    # Variables de tracking
    best_loss = float('inf')
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for videos in pbar:
            try:
                videos = processor.process_batch(videos)
                print(f"Batch shape after processing: {videos.shape}")
                videos = videos.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    reconstructed = model(videos)
                    print(f"Reconstructed shape: {reconstructed.shape}")
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
                
                # Backward y optimize
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Logging
                pbar.set_description(
                    f'Epoch {epoch} | Loss: {loss.item():.4f} | '
                    f'Recon: {recon_loss.item():.4f} | SSIM: {ssim_loss.item():.4f}'
                )
                
                # Guardar checkpoints y muestras
                if global_step % save_every == 0:
                    # Guardar si es el mejor modelo
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, results_folder / 'best_model.pt')
                    
                    # Guardar muestra de reconstrucción
                    with torch.no_grad():                        
                        save_video_tensor(
                            reconstructed[0],
                            results_folder / f'recon_video_{global_step}.mp4',
                            fps=processor.fps
                        )
                
                global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("\nOOM Error, skipping batch")
                    continue
                raise e

def TrainModelApolloVideo():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_epochs = 50
    save_every = 200
    dim_latent = 128

    # Directorios
    results_folder = Path('./results')
    results_folder.mkdir(exist_ok=True, parents=True)

    # Cargar el autoencoder pre-entrenado
    VideoAutoEncoder = EfficientVideoAutoencoder(dim_latent=dim_latent).to(device)
    autoencoder_checkpoint = torch.load(results_folder / 'best_model.pt')
    VideoAutoEncoder.load_state_dict(autoencoder_checkpoint['model_state_dict'])
    VideoAutoEncoder.eval()

    # Configurar el procesador y dataset
    processor = VideoProcessor(target_size=(240, 426))  # Mantenemos la resolución original
    dataset = VideoDataset(
        csv_path="/teamspace/studios/this_studio/datos_videos.csv",
        video_folder="/teamspace/studios/this_studio/VideoDetailCaption/Test_Videos/",
        processor=processor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )


    model = Prometheus(
        num_text_tokens=1024,
        dim_latent=dim_latent,
        modality_default_shape=(30, 30, 53),  # Ajustado para 240x426 después del encoding
        modality_encoder=VideoAutoEncoder.encoder_blocks,
        modality_decoder=VideoAutoEncoder.decoder_blocks,
        add_pos_emb=True,
        modality_num_dim=3, # 3 dimensiones para video
        transformer=dict(
            dim=768,
            depth=12,
            dim_head=12,
            heads=12,
        )
    ).to(device)

    # Optimizador y schedulers con ajustes para el modelo más grande
    optimizer = AdamW(
        model.parameters_without_encoder_decoder(), 
        lr=5e-5,  # Reducida para modelo más grande
        weight_decay=0.1,  # Aumentado para mejor regularización
        betas=(0.9, 0.999)
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=len(dataloader) * num_epochs,
        eta_min=1e-6  # Valor mínimo de learning rate
    )
    
    scaler = GradScaler()

    # Variables de tracking
    best_loss = float('inf')
    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

        for text, videos in pbar:
            try:
                # Procesar batch
                videos = processor.process_batch(videos).to(device)
                text = text.to(device)

                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    # Forward pass
                    loss = model((text, videos))

                # Backward y optimize con gradient clipping más conservador
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Logging más detallado
                if global_step % 10 == 0:  # Cada 10 pasos
                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_description(
                        f'Epoch {epoch} | Loss: {loss.item():.4f} | LR: {lr:.2e}'
                    )

                # Guardar checkpoints y muestras
                if global_step % save_every == 0:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': best_loss,
                            'global_step': global_step
                        }, results_folder / 'best_model_apollo_video.pt')

                    # Generar y guardar una muestra
                    with torch.no_grad():
                        sample_text = text[0:1]
                        generated_video = model.generate(sample_text)
                        save_video_tensor(
                            generated_video[0],
                            results_folder / f'generated_video_{global_step}.mp4',
                            fps=processor.fps
                        )

                global_step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("\nOOM Error, skipping batch")
                    continue
                raise e

        # Guardar checkpoint por época
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
            'global_step': global_step
        }, results_folder / f'checkpoint_apollo_video_epoch_{epoch}.pt')


def TrainModelApolloText():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32  # Podemos usar batch más grande para texto
    text_seq_len = 256
    num_epochs = 50
    save_every = 500
    dim_latent = 128
    VideoAutoEncoder = EfficientVideoAutoencoder(dim_latent=dim_latent).to(device)
    # Directorios
    results_folder = Path('./results')
    results_folder.mkdir(exist_ok=True, parents=True)

    # Cargar el último checkpoint del entrenamiento de video
    model = Prometheus(
        num_text_tokens=1024,
        dim_latent=128,
        modality_default_shape=(30, 30, 53),  # Mantenemos la configuración de video
        modality_encoder=VideoAutoEncoder.encoder_blocks,
        modality_decoder=VideoAutoEncoder.decoder_blocks,
        add_pos_emb=True,
        modality_num_dim=3,
        transformer=dict(
            dim=768,
            depth=12,
            dim_head=12,
            heads=12,
        )
    ).to(device)

    # Cargar checkpoint de video
    checkpoint = torch.load(results_folder / 'best_model_apollo_video.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo cargado desde época {checkpoint['epoch']} con pérdida {checkpoint['loss']}")

    # Dataset de texto
    text_dataset = load_dataset("stas/openwebtext-10k")
    train_dataset = TextDataset(text_dataset['train'], text_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    train_iter = cycle(train_loader)

    # Optimizador y scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.999)
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(train_loader),
        eta_min=1e-6
    )
    
    scaler = GradScaler()

    # Variables de tracking
    best_loss = float('inf')
    global_step = 0

    # Training loop
    total_steps = num_epochs * len(train_loader)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(num_epochs):
            for step in range(len(train_loader)):
                try:
                    model.train()
                    data = next(train_iter).to(device)

                    optimizer.zero_grad(set_to_none=True)

                    with autocast():
                        loss = model(data)

                    # Backward y optimize
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # Logging
                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_description(
                        f'Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | LR: {lr:.2e}'
                    )
                    pbar.update()

                    # Guardar checkpoints
                    if global_step % save_every == 0:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            torch.save({
                                'epoch': epoch,
                                'step': step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': best_loss,
                                'global_step': global_step
                            }, results_folder / 'best_model_apollo_text.pt')

                    global_step += 1

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print("\nOOM Error, skipping batch")
                        continue
                    raise e

            # Guardar checkpoint por época
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'global_step': global_step
            }, results_folder / f'checkpoint_apollo_text_epoch_{epoch}.pt')

    # Guardar modelo final
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': best_loss,
        'global_step': global_step
    }, results_folder / 'final_model_apollo_text.pt')

if __name__ == "__main__":
    print("Training Video Autoencoder")
    TrainAutoEncoder()
    print("Training Apollo Video Model")
    TrainModelApolloVideo()
    print("Training Apollo Text Model")
    TrainModelApolloText()
