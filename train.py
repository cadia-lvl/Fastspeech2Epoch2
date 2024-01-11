import argparse
import os
from collections import OrderedDict

# __import__('ipdb').set_trace()
import yaml
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

from utils.model import get_model, get_param_num
from utils.tools import to_device, plot_spectrograms, plot_lines, modify_length
from model import FastSpeech2Loss
from dataset import Dataset

import pdb

random_seed = 1234 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

run = wandb.init(
    # Set the project where this run will be logged
    project="FastSpeechEpoch",
    name='FastSpeechEpochResultBucket',
    # mode="offline"
)

def main(args, configs):
    preprocess_config, model_config, train_config = configs

    
    batch_size = train_config["optimizer"]["batch_size"]
    
    # Get Training dataset
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Get Testing dataset
    test_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    
    test_data_iter = iter(test_loader)
    step = 1
    pdb.set_trace()
    # Prepare model
    model, optimizer, scheduler = get_model(args, configs, device, train=True)

    # Load checkpoint
    if args.checkpoint_path:
        print(f"Loading checkpoint {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)

        # Restore the state
        modify_model_state_dict = OrderedDict([(k.split('module')[1][1:], v) for k, v in checkpoint['model_state_dict'].items()])
            
        model.load_state_dict(modify_model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        step = checkpoint['step']
    
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)
    #for p in model.parameters():
    #    print(p.numel())
    #exit(0)

    # Init path
    result_root_path = train_config["path"]["result_path"]
    result_recon_path = os.path.join(result_root_path, 'reconstruction')
    result_checkpoint_path = os.path.join(result_root_path, 'checkpoint')
    result_synthesis_path = os.path.join(result_root_path, 'synthesis')
    
    for p in [result_recon_path, result_checkpoint_path, result_synthesis_path]:
        os.makedirs(p, exist_ok=True)

    # epochlen
    epolen_bins = nn.Parameter(
                torch.linspace(0.0024999999999995026, 0.02400000000000002, 257 - 1),
                requires_grad=False,
    ).to(device)

    # Training
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    vis_step = train_config["step"]["vis_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                '''
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max_text_len,   
                    mels,
                    phases,
                    acoustic_lens,
                    max_acoustic_len,   
                    epochdurs,
                    epochlens
                '''
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                
                (
                    total_loss,
                    mel_loss_l1,
                    mel_loss_l2,
                    phase_loss_l1,
                    phase_loss_l2,
                    duration_loss_l1,
                    duration_loss_l2,
                    length_loss_ce
                ) = losses
                

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Log the loss
                if step % log_step == 0:
                    #losses_keys = ['Total Loss', 'Complex Loss', 'Duration L1 Loss', 'Duration L2 Loss', 'Epoch Len CE']
                    losses_keys = ['Total Loss', 'Mel Loss L1', 'Mel Loss L2', 'Phase Loss L1', 'Phase Loss L2', 'Duration Loss L1', 'Duration Loss L2', 'Epoch Len EC']
                    losses_report = {l_key: l.item() for l_key, l in zip(losses_keys, losses)}                    
                    wandb.log(losses_report, step=step)

                # Visualize the reconstruction
                if step % vis_step == 0:

                    vis_target_mel = batch[6][0]
                    vis_target_phase = batch[7][0]
                    vis_acoustic_len = batch[8][0].item()
                    vis_epochdur = batch[-2][0].reshape(-1) # torch.Size([77])
                    vis_epochlen = batch[-1][0].reshape(-1) # torch.Size([1615])
                    vis_epochlen_bucket = torch.bucketize(vis_epochlen, epolen_bins)
                    
                    vis_predict_mel = output[1][0].transpose(0,1)
                    vis_predict_phase = output[2][0].transpose(0,1)
                    vis_predict_epodur = torch.exp(output[0][0].reshape(-1))
                    vis_predict_epolen = torch.argmax(output[3][0], dim=1)
                    
                    # vis_predict_epolen = torch.exp(vis_predict_epolen) / 10.0
                    
                    vis_target_mel = vis_target_mel[:, :vis_acoustic_len].cpu().detach().numpy() 
                    vis_target_phase = vis_target_phase[:, :vis_acoustic_len].cpu().detach().numpy() 
                    vis_predict_mel = vis_predict_mel[:, :vis_acoustic_len].cpu().detach().numpy() 
                    vis_predict_phase = vis_predict_phase[:, :vis_acoustic_len].cpu().detach().numpy() 
                    
                    vis_epochdur = vis_epochdur.cpu().detach().numpy()
                    vis_predict_epodur = vis_predict_epodur.cpu().detach().numpy()
                    
                    vis_epochlen_bucket = vis_epochlen_bucket[:vis_acoustic_len].cpu().detach().numpy()
                    vis_predict_epolen = vis_predict_epolen[:vis_acoustic_len].cpu().detach().numpy()
                    
                    os.makedirs(os.path.join(result_recon_path, f"{step}"), exist_ok=True)
                    
                    plot_spectrograms(vis_target_mel, vis_predict_mel, os.path.join(result_recon_path, f"{step}", 'mel.png'))
                    plot_spectrograms(vis_target_phase, vis_predict_phase, os.path.join(result_recon_path, f"{step}", 'phase.png'))
                    
                    plot_lines(vis_epochdur, vis_predict_epodur, os.path.join(result_recon_path, f"{step}", 'epochdur.png'))
                    plot_lines(vis_epochlen_bucket, vis_predict_epolen, os.path.join(result_recon_path, f"{step}", 'epochlen.png'))
                    
                    np.save(os.path.join(result_recon_path, f"{step}", 'predict_mel.npy'), vis_predict_mel)
                    np.save(os.path.join(result_recon_path, f"{step}", 'predict_phase.npy'), vis_predict_phase)
                    np.save(os.path.join(result_recon_path, f"{step}", 'predict_epodur.npy'), vis_predict_epodur)
                    np.save(os.path.join(result_recon_path, f"{step}", 'predict_epolen.npy'), vis_predict_epolen)
                    

                if step % synth_step == 0:
                    model.eval()
                    
                    try:
                        test_batchs = next(test_data_iter)
                    except:
                        test_data_iter = iter(test_loader)
                        test_batchs = next(test_data_iter)
                    
                    test_batch = test_batchs[0]
                    test_batch = to_device(test_batch, device)

                    test_output = model(test_batch[2], test_batch[3], test_batch[4], test_batch[5])
                    
                    vis_target_mel = test_batch[6][0]
                    vis_target_phase = test_batch[7][0]
                    vis_acoustic_len = test_batch[8][0].item()
                    vis_epochdur = test_batch[-2][0].reshape(-1)
                    vis_epochlen = test_batch[-1][0].reshape(-1) # torch.Size([1615])
                    vis_epochlen_bucket = torch.bucketize(vis_epochlen, epolen_bins)
                    
                    vis_predict_mel = test_output[1][0].transpose(0,1)
                    vis_predict_phase = test_output[2][0].transpose(0,1)
                    vis_predict_epodur = torch.exp(test_output[0][0].reshape(-1))
                    vis_predict_epolen = torch.argmax(test_output[3][0], dim=1)
                    
                    # vis_predict_epolen = torch.exp(vis_predict_epolen) / 1000.0
                    
                    vis_max_acoustic_len = max(vis_acoustic_len, vis_predict_mel.shape[-1])
                    
                    vis_target_mel = vis_target_mel.cpu().detach().numpy() 
                    vis_target_phase = vis_target_phase.cpu().detach().numpy() 
                    vis_target_mel = modify_length(vis_target_mel, maxlen=vis_max_acoustic_len)
                    vis_target_phase = modify_length(vis_target_phase, maxlen=vis_max_acoustic_len)
                    
                    vis_predict_mel = vis_predict_mel.cpu().detach().numpy() 
                    vis_predict_phase = vis_predict_phase.cpu().detach().numpy() 
                    vis_predict_mel = modify_length(vis_predict_mel, maxlen=vis_max_acoustic_len)
                    vis_predict_phase = modify_length(vis_predict_phase, maxlen=vis_max_acoustic_len)
                    
                    vis_epochdur = vis_epochdur.cpu().detach().numpy()
                    vis_predict_epodur = vis_predict_epodur.cpu().detach().numpy()
                    
                    vis_epochlen_bucket = vis_epochlen_bucket.cpu().detach().numpy()
                    vis_predict_epolen = vis_predict_epolen.cpu().detach().numpy()
                    
                    os.makedirs(os.path.join(result_synthesis_path, f"{step}") ,exist_ok=True)
                    
                    plot_spectrograms(vis_target_mel, vis_predict_mel, os.path.join(result_synthesis_path, f"{step}", 'mel.png'))
                    plot_spectrograms(vis_target_phase, vis_predict_phase, os.path.join(result_synthesis_path, f"{step}", 'phase.png'))
                    
                    plot_lines(vis_epochdur, vis_predict_epodur, os.path.join(result_synthesis_path, f"{step}", 'epochdur.png'))
                    plot_lines(vis_epochlen_bucket, vis_predict_epolen, os.path.join(result_synthesis_path, f"{step}", 'epochlen.png'))
                    
                    np.save(os.path.join(result_synthesis_path, f"{step}", 'predict_mel.npy'), vis_predict_mel)
                    np.save(os.path.join(result_synthesis_path, f"{step}", 'predict_phase.npy'), vis_predict_phase)
                    np.save(os.path.join(result_synthesis_path, f"{step}", 'predict_epodur.npy'), vis_predict_epodur)
                    np.save(os.path.join(result_synthesis_path, f"{step}", 'predict_epolen.npy'), vis_predict_epolen)
                    
                    model.train()
                    
                

                if step % save_step == 0:
                    checkpoint = {
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(result_checkpoint_path, f"{step}.ckpt"))
                    print()
                    

                if step == total_step:
                    quit()
                    
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='')
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
