# -*- coding: UTF-8 -*-
# Local packages:
import os
import pickle
from typing import Dict, Union

# 3rd party packages:
import numpy as np
from tqdm import tqdm,trange
import torch 
from torch.utils.tensorboard import SummaryWriter

# TODO: Implement Neptune logger

# personal packages:
from model.dataset import TimeGAN_Dataset

def embedding_trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        e_opt: torch.optim.Optimizer,
        r_opt: torch.optim.Optimizer,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
):
    """
    Training loop for mebedding and recovery functions.
    Args:
        model (torch.nn.Module): The model to train
        dataloader (torch.utils.data.DataLoader): The dataloader to use
        e_opt (torch.optim.Optimizer): The optimizer for the embedding function
        r_opt (torch.optim.Optimizer): The optimizer for the recovery function
        args (Dict): The model/training configuration
        writer (Union[torch.utils.tensorboard.SummaryWriter, type(None)], optional): The tensorboard writer to use. Defaults to None.
    """
    logger = trange(args.emb_epochs, desc =f"Epoch:0, Loss:0")
    for epoch in logger:
        for X_mb,T_mb in dataloader:

            #reset gradients
            model.zero_grad()

            #forward pass
            _,E_loss0,E_loss_T0 = model(X=X_mb,T=T_mb,Z=None,obj="autoencoder")
            loss = np.sqrt(E_loss_T0.item())

            #backward pass
            E_loss0.backward()

            #update weights
            e_opt.step()
            r_opt.step()

        # Log loss for final batch of each epochs
        logger.set_description(f"Epoch:{epoch}, Loss:{loss:.4f}")
        if writer:
            writer.add_scalar("Embedding/Loss:",loss,epoch)
            writer.flush()

def supervisor_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
):
    """
    The training loop for the supervisor function
    Args:
        model (torch.nn.Module): The model to train
        dataloader (torch.utils.data.DataLoader): The dataloader to use
        s_opt (torch.optim.Optimizer): The optimizer for the supervisor function
        g_opt (torch.optim.Optimizer): The optimizer for the generator function
        args (Dict): The model/training configuration
        writer (Union[torch.utils.tensorboard.SummaryWriter, type(None)], optional): The tensorboard writer to use. Defaults to None.
    """
    logger = trange(args.sup_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            S_loss = model(X=X_mb, T=T_mb, Z=None, obj="supervisor")

            # Backward Pass
            S_loss.backward()
            loss = np.sqrt(S_loss.item())

            # Update model parameters
            s_opt.step()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar(
                "Supervisor/Loss:",loss,epoch)
            writer.flush()

def joint_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    d_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None, 
):
    """
    The training loop for training the model altogether
    Args:
        model (torch.nn.Module): The model to train
        dataloader (torch.utils.data.DataLoader): The dataloader to use
        e_opt (torch.optim.Optimizer): The optimizer for the embedding function
        r_opt (torch.optim.Optimizer): The optimizer for the recovery function
        s_opt (torch.optim.Optimizer): The optimizer for the supervisor function
        g_opt (torch.optim.Optimizer): The optimizer for the generator function
        d_opt (torch.optim.Optimizer): The optimizer for the discriminator function
        args (Dict): The model/training configuration
        writer (Union[torch.utils.tensorboard.SummaryWriter, type(None)], optional): The tensorboard writer to use. Defaults to None.
    """
    logger = trange(
        args.sup_epochs, 
        desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
    )
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            ## Generator Training
            for _ in range(2):
                # Random Generator
                Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

                # Forward Pass (Generator)
                model.zero_grad()
                G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                # Update model parameters
                g_opt.step()
                s_opt.step()

                # Forward Pass (Embedding)
                model.zero_grad()
                E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())
                
                # Update model parameters
                e_opt.step()
                r_opt.step()

            # Random Generator
            Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

            ## Discriminator Training
            model.zero_grad()
            # Forward Pass
            D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")

            # Check Discriminator loss
            if D_loss > args.dis_thresh:
                # Backward Pass
                D_loss.backward()

                # Update model parameters
                d_opt.step()
            D_loss = D_loss.item()

        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
        )
        if writer:
            writer.add_scalar(
                'Joint/Embedding_Loss:',E_loss,epoch)
            writer.add_scalar(
                'Joint/Generator_Loss:',G_loss,epoch)
            writer.add_scalar('Joint/Discriminator_Loss:',D_loss,epoch)
            writer.flush()

def timegan_trainer(model,loaded_data,args):
    """
    The trainign procedure for TimeGAN.
    Args:
        model (torch.nn.module): The model that generates synthetic data
        loaded_data(pandas.DataFrame): The data to train on, including data and time
        args (Dict): The model/training configuration
    Returns:
        generated_data (np.array): The synthetic data generated by the model
    """
    dataset = TimeGAN_Dataset(loaded_data["data"],loaded_data["time"])
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    model.to(args.device)

    #initialize optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.lr)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.lr)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.lr)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.lr)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)

    #initialize tensorboard writer
    writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))

    print("\nStart Embedding Network Training")
    embedding_trainer(model=model, dataloader=dataloader, e_opt=e_opt, r_opt=r_opt, args=args, writer=writer)

    print("\nStart Training with Supervised Loss Only")
    supervisor_trainer(model=model, dataloader=dataloader, s_opt=s_opt,g_opt=g_opt, args=args, writer=writer)

    print("\nStart Joint Training")
    joint_trainer(model=model, dataloader=dataloader, e_opt=e_opt, r_opt=r_opt, s_opt=s_opt, g_opt=g_opt, d_opt=d_opt, args=args, writer=writer)


    #save model,args, and hyperparameters
    torch.save(args,f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(),f"{args.model_path}/model.pt")
    print(f"Model saved to {args.model_path}")

    def timegan_generator(model,T,args):
        """
        The interference procedure for TimeGAN.

        Args:
            model (torch.nn.module): The model that generates synthetic data
            T (List[int]): The time to generate data for
            args (Dict): The model/training configuration
        returns:
            generated_data (np.array): The synthetic data generated by the model
        """

        #load model
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model not found at {args.model_path}")
        model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))

        print("\nStart Generating Synthetic Data")
        #Initialize model to evaluation mode and run without gradients
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            # Random Generator
            Z = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

            # Forward Pass (Generator)
            generated_data = model(X=None, T=T, Z=Z, obj="inference")
        return generated_data.numpy()
