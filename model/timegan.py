# -*- coding: UTF-8 -*-
# TimeGan implementation
# Path: timeGAN_ex.py
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingNetwork(nn.Module):
    """
    The embedding network (encoder) that maps the input data to a latent space.
    """
    def __init__(self,args):
        super(EmbeddingNetwork, self).__init__()
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self. max_seq_len = args.max_seq_len

        #Embedder Architecture
        self.emb_rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.emb_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = nn.Sigmoid()

        
        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L61

        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(1)

            for name, param in self.emb_linear.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self,X,T):
        """Forward pass of the embedding features from original space to latent space.
        Args:
            X: Input time series feature (B x S x F)
            T: INput temporal information (B)
        Returns:
            H: latent space embeddings (B x S x H)
        """
        # Dynamic RNN input for ignoring paddings

        X_pack = nn.utils.rnn.pack_padded_sequence(
            input =X,
            lengths=T,
            batch_first=True,
            enforce_sorted=False,
        )

        # 128*100*71
        H_o,H_t = self.emb_rnn(X_pack)

        #pad RNN output back to sequence length

        H_o,T = nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        #128*100*10
        logits = self.emb_linear(H_o)
        H = self.emb_sigmoid(logits)

        return H
    
class RecoveryNetwork(nn.Module):
    """The recovery network (decoder) for TimeGAN
    """
    def __init__(self,arg):
        super(RecoveryNetwork, self).__init__()
        self.hidden_dim = arg.hidden_dim
        self.feature_dim = arg.feature_dim
        self.num_layers = arg.num_layers
        self.padding_value = arg.padding_value
        self.max_seq_len = arg.max_seq_len

        #Recovery Architecture
        self.rec_rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.rec_linear = nn.Linear(self.hidden_dim, self.feature_dim)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614

        with torch.no_grad():
            for name,param in self.rec_rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name,param in self.rec_linear.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
        
    def forward(self,H,T):
        """ Forward pass of the recovery features from latent space to original space.
        Args:
            H: latent representation (B x S x E)
            T: input temporal information (B)
        Returns:
            X_tilde: recovered features (B x S x F)
        """
        #Dynamic RNN input for ignoring paddings
        H_pack = nn.utils.rnn.pack_padded_sequence(
            input = H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False,
        )
        #128 x 100 x 10
        H_o,H_t = self.rec_rnn(H_pack)
        #pad RNN output back to sequence length
        H_o,T = nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )
        #128 x 100 x 71
        X_tilde = self.rec_linear(H_o)
        return X_tilde

class SupervisorNetwork(nn.Module):
        """The supervisor network for TimeGAN
        """
        def __init__(self,args):
            super(SupervisorNetwork,self).__init__()
            self.hidden_dim = args.hidden_dim
            self.num_layers = args.num_layers
            self.padding_value = args.padding_value
            self.max_seq_len = args.max_seq_len

            #supervisor architecture
            self.sup_rnn = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers-1,
                batch_first=True,
            )
            self.sup_linear = nn.Linear(self.hidden_dim,self.hidden_dim)
            self.sup_sigmoid = nn.Sigmoid()
             # Init weights
            # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
            # Reference: 
            # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
            # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
            with torch.no_grad():
                for name, param in self.sup_rnn.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(1)
                    elif 'bias_hh' in name:
                        param.data.fill_(0)
                for name, param in self.sup_linear.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
        def forward(self,H,T):
            """Forward pass for the supervisor for predicting next step
            Args:
                H: latent representation (B x S x E)
                T: input temporal information (B)
            Returns:
                H_hat: predicted next step data (B x S x E)
            """

            #Dynamic RNN input for ignoring paddings
            H_pack = nn.utils.rnn.pack_padded_sequence(
                input = H,
                lengths=T,
                batch_first=True,
                enforce_sorted=False,
            )

            H_o,H_t = self.sup_rnn(H_pack)
            #pad RNN output back to sequence length
            H_o,T = nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len,
            )
            logits = self.sup_linear(H_o)
            H_hat = self.sup_sigmoid(logits)
            return H_hat

class GeneratorNetwork(nn.Module):
    """The generator network for TimeGAN
    """
    def __init__(self,args):
        super(GeneratorNetwork,self).__init__()
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        #Generator Architecture
        self.gen_rnn = nn.GRU(
            input_size=self.Z_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.gen_linear = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.gen_sigmoid = nn.Sigmoid()
                # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
        
    def forward(self,Z,T):
        """ Takes in random noise (features) and generates synthetic features within the last latent space
        Args:
            Z: input random noise (B x S x Z)
            T: input temporal information (B)
        Returns:
            H: embeddings (B x S x E)
        """
        #Dynamic RNN input for ignoring paddings
        Z_pack = nn.utils.rnn.pack_padded_sequence(
            input = Z,
            lengths=T,
            batch_first=True,
            enforce_sorted=False,
        )

        # 128*100*71
        H_o,H_t = self.gen_rnn(Z_pack)

        #pad RNN output back to sequence length

        H_o,T = nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        #128*100*10
        logits = self.gen_linear(H_o)
        H = self.gen_sigmoid(logits)

        return H

class DiscriminatorNetwork(nn.Module):
    """The discriminator network for TimeGAN
    """
    def __init__(self,args):
        super(DiscriminatorNetwork,self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        #Discriminator Architecture
        self.dis_rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dis_linear = nn.Linear(self.hidden_dim,1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, H, T):
        """ Forward pass for predicting if the data is real or synthetic
        
        Args:
            H: latent representation (B x S x E)
            T: input temporal information (B)
        Returns:
        logits: prediction logits(B x S x 1)
        """
        # dynamic RNN input for ignoring paddings
        H_pack = nn.utils.rnn.pack_padded_sequence(
            input = H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False,
        )

        # 128*100*10
        H_o,H_t = self.dis_rnn(H_pack)

        # pad RNN output back to sequence length
        H_o,T = nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )

        logits = self.dis_linear(H_o).squeeze(-1)
        return logits

class TimeGAN(nn.Module):
    """ Implementation of TimeGan (Yoon et al., 2019) using PyTorch
    
    Reference:
        - Yoon, J., Jarret, D., van der Schaar, M. (2019). Time-series Generative Adversarial Networks. (https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)
        - https://github.com/jsyoon0823/TimeGAN
    """
    def __init__(self,args):
        super(TimeGAN,self).__init__()
        self.device =args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size

        self.embedder = EmbeddingNetwork(args)
        self.recovery = RecoveryNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)
        self.supervisor = SupervisorNetwork(args)

    def _recovery_forward(self, X, T):
        """ The embedding network forward pass and the embedder network loss
        Args:
            X: input features
            T: input temporal information
        Returns:
            E_loss: the reconstruction loss
            X_tilde: the reconstructed features
        """

        # FOrward pass
        H = self.embedder(X,T)
        X_tilde = self.recovery(H,T)

        #for Joint training
        H_hat_supervise = self.supervisor(H,T)
        G_loss_S = F.mse_loss(
            H_hat_supervise[:,:-1,:],
            H[:,1:,:],
        ) #Teacher forcing next output

        #Reconstruction loss
        E_loss_T0 = F.mse_loss(X_tilde,X)
        E_loss0 = 10*torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1*G_loss_S
        return E_loss, E_loss0,E_loss_T0
    def _supervisor_forward(self, X, T):
        """ The supervisor training forward pass
        Args:
            X: original input features
            T: input temporal information
        Returns:
            S_loss: the supervisor's loss
        """
        #supervisor forward pass
        H = self.embedder(X,T)
        H_hat_supervise = self.supervisor(H,T)

        #supervised loss
        S_loss = F.mse_loss(
            H_hat_supervise[:,:-1,:],
            H[:,1:,:],
        ) #Teacher forcing next output
        return S_loss
    def _discriminator_forward(self, X, T, Z, gamma=1):
        """ The discriminator forward pass and adversarial loss
        Args:
            X: input features
            T: input temporal information
            Z: input noise
            gamma: the weight for the adversarial loss
        Returns:
            D_loss: adversarial loss
        """
        #Real
        H = self.embedder(X, T).detach()

        #generator
        E_hat = self.generator(Z,T).detach()
        H_hat = self.supervisor(E_hat,T).detach()
        
        #forward pass
        Y_real = self.discriminator(H,T)        #Encode original data
        Y_fake = self.discriminator(H_hat,T)    #Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat,T)  #Output of generator

        D_loss_real = F.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
        D_loss_fake = F.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = F.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss
    
    def _generator_forward(self, X, T, Z, gamma=1):
        """ The generator forward pass
        Args:
            X: original input features
            T: input temporal information
            Z: input noise for the generator
            gamma: the weight for the adversarial loss
        Returns:
            G_loss: the generator loss
        """
        #supervisor forward pass
        H = self.embedder(X,T)
        H_hat_supervise = self.supervisor(H,T)

        #generator forward pass
        E_hat = self.generator(Z,T)
        H_hat = self.supervisor(E_hat,T)

        #synthetic data generated
        X_hat = self.recovery(H_hat,T)

        #generator loss
        #Adversarial loss
        Y_fake = self.discriminator(H_hat,T)        #Output of supervisor
        Y_fake_e = self.discriminator(E_hat,T)      #Output of generator

        G_loss_U = F.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = F.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

        #Supervised loss
        G_loss_S = F.mse_loss(
            H_hat_supervise[:,:-1,:],
            H[:,1:,:],
        ) #Teacher forcing next output

        #Two moments losses
        G_loss_V1 = torch.mean(
            torch.abs(torch.sqrt(X_hat.var(dim=0,unbiased=False)+1e-6) - torch.sqrt(X.var(dim=0,unbiased=False)+1e-6))
        )
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))
        G_loss_V = G_loss_V1 + G_loss_V2
        
        #sum of losses
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
    
        return G_loss
    
    def _inference(self, Z,T):
        """ Inference for generating synthetic data
        Args:
            Z: input noise
            T: temporal information
        Returns:
            X_hat: the generated data
        """

        #generator forward pass
        E_hat = self.generator(Z,T)
        H_hat = self.supervisor(E_hat,T)

        #synthetic data generated
        X_hat = self.recovery(H_hat,T)
        return X_hat

    def forward(self,X,T,Z, obj, gamma=1):
        """
        Args:
            X: input features (B,H,F)
            T: The temporal information (B)
            Z: the sampled noise (B,H,Z)
            obj: the network to be trained ('autoencoder','supervisor','generator','discriminator')
            gamma: loss hyperparameter
        Returns:
            loss: loss for the forward pass
            X_hat: the generated data
        """

        #Move variables to device
        if obj !='inference':
            if X is None:
                raise ValueError('X cannot be empty')
            
            X = torch.FloatTensor(X)
            X = X.to(self.device)

        if Z is not None:
            Z = torch.FloatTensor(Z)
            Z = Z.to(self.device)
        
        if obj == 'autoencoder':
            #embedder and recovery forward
            loss = self._recovery_forward(X,T)
        elif obj == 'supervisor':
            loss = self._supervisor_forward(X,T)
        elif obj == 'generator':
            if Z is None:
                raise ValueError('Z cannot be empty')
            loss = self._generator_forward(X,T,Z,gamma)
        elif obj == 'discriminator':
            if Z is None:
                raise ValueError('Z cannot be empty')
            loss = self._discriminator_forward(X,T,Z,gamma)
            return loss
        elif obj == 'inference':
            X_hat = self._inference(Z,T)
            X_hat = X_hat.cpu.detach()

            return X_hat
        else:
            raise ValueError('obj must be autoencoder, supervisor, generator or discriminator')
        return loss
        
        

                

