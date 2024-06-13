import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.visualization import *
import utils.visualize_function as vf



@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):

    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg,val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size,1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size,1)

    

   
    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',    # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',            # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
    collate_fn=train_set.collate_fn)


    # Visualize the data
    for batch_dict in enumerate(train_loader):
        inputs = batch_dict[1]['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'],inputs['obj_trajs_mask'] ,inputs['map_polylines']
        roads_mask = inputs['map_polylines_mask'] 
        agents_in = torch.cat([agents_in[...,:2],agents_mask.unsqueeze(-1)],dim=-1)
        ego_in = torch.gather(inputs['obj_trajs'], 1, inputs['track_index_to_predict'].view(-1,1,1,1).repeat(1,1,*inputs['obj_trajs'].shape[-2:])).squeeze(1)

        # Call the visualization function to obtain the matplotlib figure
        for i in range(batch_dict[1]['batch_size']):
            sid= batch_dict[1]['input_dict'].get('scenario_id')[i]


            plt_fig= check_loaded_data(batch_dict[1]['input_dict'], index=i, highlight_idx= 2)

            plt_fig.save(f'/home/omar/MotionNetAO/img/batch_{i}_{sid}.png')

        
   


    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
    collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger= None if cfg.debug else WandbLogger(project="motionnet", name=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs
    )

    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=cfg.ckpt_path)
    #captured_outputs = trainer.callbacks[0].outputs  # From the OutputCaptureCallback


if __name__ == '__main__':
    train()

