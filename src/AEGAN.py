import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input, Dropout, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import Conv2D, UpSampling2D, Reshape, Flatten, MaxPooling2D, LayerNormalization
from tensorflow.keras.models import Sequential
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow.keras.optimizers.legacy as opt
import matplotlib.pyplot as plt
import time
import os
from IPython.display import clear_output

class AEGAN():
  def __init__(self, 
               input_dim=(0, 0, 0), 
               discriminator_x_conv_filters=[], 
               discriminator_x_conv_kernels=[], 
               discriminator_x_conv_strides=[],
               discriminator_z_neurons=[], 
               generator_conv_filters=[], 
               generator_conv_kernels=[], 
               generator_conv_strides=[],
               encoder_conv_filters=[], 
               encoder_conv_kernels=[], 
               encoder_conv_strides=[],
               z_dim=16,
               learning_rate=0.0002):
    self.img_rows = input_dim[0]
    self.img_cols = input_dim[1]
    self.img_channels = input_dim[2]
    self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
    self.dx_conv_fil = discriminator_x_conv_filters
    self.dx_conv_ker = discriminator_x_conv_kernels
    self.dx_conv_str = discriminator_x_conv_strides
    self.dz_neurons = discriminator_z_neurons
    self.g_conv_fil = generator_conv_filters
    self.g_conv_ker = generator_conv_kernels
    self.g_conv_str = generator_conv_strides
    self.e_conv_fil = encoder_conv_filters
    self.e_conv_ker = encoder_conv_kernels
    self.e_conv_str = encoder_conv_strides
    self.vector_dim = z_dim
    self.lr = learning_rate
  
  def build(self, use_dropout=False):
    # Build and compile individual nets
    optimizer = opt.Adam(self.lr, 0.5)

    self.D_x = self._build_discriminator_X(use_dropout)
    self.D_x.compile(loss='binary_crossentropy', optimizer=optimizer)
    self.D_z = self._build_discriminator_Z()
    self.D_z.compile(loss='binary_crossentropy', optimizer=optimizer)
    self.G = self._build_generator()
    self.E = self._build_encoder()

    self.G.summary()
    self.D_x.summary()
    self.E.summary()
    self.D_z.summary()

    #------------
    # Build graph
    #------------

    # Input image
    x = Input(shape=self.img_shape)
    # Input latent vector
    z = Input(shape=(self.vector_dim,))

    # Fake generated image
    x_hat = self.G(z)
    # Fake latent vector
    z_hat = self.E(x)
    # Generated image from fake latent vector
    x_tilde = self.G(z_hat)
    # Generated latent vector from fake image
    z_tilde = self.E(x_hat)

    # Prediction of x_hat
    pred_x_hat = self.D_x(x_hat)
    # Prediction of x_tilde
    pred_x_tilde = self.D_x(x_tilde)
    # Prediction of z_hat
    pred_z_hat = self.D_z(z_hat)
    # Prediction of z_tilde
    pred_z_tilde = self.D_z(z_tilde)

    # For the combined model, only generators are trained
    self.D_x.trainable = False
    self.D_z.trainable = False

    self.combined = keras.Model(inputs=[x,z], outputs=[x_tilde, z_tilde, pred_x_hat, pred_x_tilde, pred_z_hat, pred_z_tilde])
    self.combined.compile(loss=['mae', 'mse', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], 
                          loss_weights=[10,5,1,1,1,1],
                          optimizer=optimizer)
  
  # Method for building the generator
  def _build_generator(self):
    starting_dim = self.img_rows
    for _ in range(len(self.g_conv_fil)-1):
      starting_dim = starting_dim // 2
    n_nodes = starting_dim*starting_dim*self.g_conv_fil[0]
    
    generator = Sequential(name='Generator')
    generator.add(Input((self.vector_dim,), name='G_Input'))
    generator.add(Dense(n_nodes, kernel_initializer=RandomNormal(stddev=0.02), name='G_Expansion'))
    generator.add(LayerNormalization())
    generator.add(Activation('relu'))
    generator.add(Reshape((starting_dim,starting_dim,self.g_conv_fil[0])))
    
    for i in range(len(self.g_conv_fil)-2):
        generator.add(UpSampling2D((2, 2), name='G_Upsampling_'+str(i+1)))
        generator.add(Conv2D(self.g_conv_fil[i], kernel_size=self.g_conv_ker[i], strides=self.g_conv_str[i], padding='same', name='G_Conv_'+str(i+1)))
        generator.add(LayerNormalization())
        generator.add(Activation('relu'))
    
    generator.add(UpSampling2D(size=(2,2), name='G_LastUpsampling'))
    generator.add(Conv2D(self.img_channels, kernel_size=(3,3), padding='same', activation='tanh', name='G_Oputput'))
    return generator

  # Method for building the encoder
  def _build_encoder(self):
    encoder = Sequential(name='Encoder')
    encoder.add(Input(self.img_shape, name='E_Input'))
    
    for i in range(len(self.e_conv_fil)-1):
        encoder.add(Conv2D(self.e_conv_fil[i], kernel_size=self.e_conv_ker[i], strides=self.e_conv_str[i], padding='same', name='E_Conv_'+str(i+1)))
        encoder.add(LayerNormalization())
        encoder.add(Activation('relu'))
        encoder.add(MaxPooling2D(2))
    
    encoder.add(Flatten())
    encoder.add(Dense(self.e_conv_fil[-1], name='E_Dense'))
    encoder.add(Activation('relu'))
    encoder.add(Dense(self.vector_dim, activation='linear', name='E_Output'))
    return encoder

  # Method for building the image discriminator
  def _build_discriminator_X(self, dropout=False):
    discriminator = Sequential(name='Discriminator_X')
    discriminator.add(Input(self.img_shape, name='DX_Input'))
    discriminator.add(GaussianNoise(0.01))
    
    for i in range(len(self.dx_conv_fil)-1):
        discriminator.add(Conv2D(self.dx_conv_fil[i], kernel_size=self.dx_conv_ker[i], strides=self.dx_conv_str[i], padding='same', name='DX_Conv_'+str(i+1)))
        discriminator.add(LayerNormalization())
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(MaxPooling2D(2))
    
    discriminator.add(Flatten())
    if dropout:
        discriminator.add(Dropout(0.3))
    discriminator.add(Dense(self.dx_conv_fil[-1], name='DX_Dense'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1, activation='sigmoid', name='DX_Output'))
    return discriminator

  # Method for building the latent vector discriminator
  def _build_discriminator_Z(self):
    discriminator = Sequential(name='Discriminator_Z')
    discriminator.add(Input(shape=(self.vector_dim,), name='DZ_Input'))
    discriminator.add(GaussianNoise(0.01))
    discriminator.add(Dense(self.dz_neurons[0], name='DZ_Dense_1'))
    discriminator.add(GaussianDropout(0.05))
    discriminator.add(LayerNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(self.dz_neurons[1], name='DZ_Dense_2'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1, activation='sigmoid', name='DZ_Output'))
    return discriminator

  #-------
  # Generate a batch of new fake images
  #-------

  def _generate(self, save_path, amount=25, epoch=0):
    generated_images = self.G.predict(np.random.normal(0, 1, size=[amount, self.vector_dim]))

    def cancel_preprocessing(img):
      img_p = (img+1)/2
      #print(img_p.max(), img_p.min())
      return img_p

    images=[]
    for i in range(amount):
      images.append(generated_images[i])

    fig = plt.figure(figsize=(12,12))
    grid = ImageGrid(fig, 111, nrows_ncols=(5,5), axes_pad=0.1)

    for ax, im in zip(grid, images):
      ax.axis('off')
      ax.imshow(cancel_preprocessing(im))

    filename = 'generated_plot_e%03d.png' % (epoch+1)
    path = os.path.join(save_path, filename)
    plt.savefig(path)
    plt.show()
    plt.close()
  
  #-------------
  # Save Models
  #-------------
  def save(self, path):
    self.G.save(os.path.join(path, 'aegan_Generator.keras'))
    self.E.save(os.path.join(path, 'aegan_Encoder.keras'))
    self.D_x.save(os.path.join(path, 'aegan_DiscriminatorImage.keras'))
    self.D_z.save(os.path.join(path, 'aegan_DiscriminatorLatent.keras'))
    self.combined.save(os.path.join(path, 'aegan_Combined.keras'))

  #-------
  # Train
  #-------
  
  # Training method
  def train(self, dataflow, save_path, epochs=100, verbose=True, show_predictions=True):
    steps_per_epoch = len(dataflow)
    print('Training Started')
    start_time = time.time()
    history = {'Rx':[], 'Rz':[], 'Dx':[], 'Dz':[], 'Gx':[], 'Gz':[]}
    for epoch in range(epochs):
      print('-'*15, f'EPOCH {epoch+1}/{epochs}', '-'*15)
      running_loss_rx = 0
      running_loss_rz = 0
      running_loss_dx = 0
      running_loss_dz = 0
      running_loss_gx = 0
      running_loss_gz = 0
      
      for step in range(steps_per_epoch):
        batch, _ = dataflow[step]

        # Create labels for discriminator
        real_labels_d = np.ones((len(batch),1))*0.95
        fake_labels_d = np.ones((len(batch),1))*0.05
        # Create labels for generator
        labels_g = np.ones((len(batch),1))

        # Train discriminators
        x1 = batch
        x2 = batch
        x_hat = self.G.predict(np.random.normal(0, 1, size=[len(batch), self.vector_dim]))
        x_tilde = self.G.predict(self.E.predict(batch))
        
        self.D_x.trainable=True
        running_loss_dx += self.D_x.train_on_batch(x1, real_labels_d)
        running_loss_dx += self.D_x.train_on_batch(x_hat, fake_labels_d)
        running_loss_dx += self.D_x.train_on_batch(x2, real_labels_d)
        running_loss_dx += self.D_x.train_on_batch(x_tilde, fake_labels_d)
        self.D_x.trainable=False
        del x_tilde, x_hat, x1, x2

        z1 = np.random.normal(0, 1, size=[len(batch), self.vector_dim])
        z2 = np.random.normal(0, 1, size=[len(batch), self.vector_dim])
        z_hat = self.E.predict(batch)
        z_tilde = self.E.predict(self.G.predict(np.random.normal(0, 1, size=[len(batch), self.vector_dim])))

        self.D_z.trainable=True
        running_loss_dz += self.D_z.train_on_batch(z1, real_labels_d)
        running_loss_dz += self.D_z.train_on_batch(z_hat, fake_labels_d)
        running_loss_dz += self.D_z.train_on_batch(z2, real_labels_d)
        running_loss_dz += self.D_z.train_on_batch(z_tilde, fake_labels_d)
        self.D_z.trainable=False
        del z_tilde, z_hat, z1, z2

        # Train generators
        images = batch
        latent = np.random.normal(0, 1, size=[len(batch), self.vector_dim])
        losses = self.combined.train_on_batch(
            [images, latent],
            [images, latent, labels_g, labels_g, labels_g, labels_g]
        )
        (_, loss_rx, loss_rz, loss_dx_g_z, loss_dx_g_e_x, loss_dz_e_x, loss_dz_e_g_z, ) = losses

        running_loss_rx += loss_rx
        running_loss_rz += loss_rz
        running_loss_gx += (loss_dx_g_e_x + loss_dx_g_z) / 2
        running_loss_gz += (loss_dz_e_g_z + loss_dz_e_x) / 2
        
        if verbose:
          t = int(time.time() - start_time)
          print(f"Epoch: [{epoch+1}/{epochs}], Training step: [{(step+1)}/{steps_per_epoch}]: "
                f"Gx={running_loss_gx/(step+1):.4f}; "
                f"Gz={running_loss_gz/(step+1):.4f}; "
                f"Dx={running_loss_dx/(step+1):.4f}; "
                f"Dz={running_loss_dz/(step+1):.4f}; "
                f"Rx={running_loss_rx/(step+1):.4f}; "
                f"Rz={running_loss_rz/(step+1):.4f}; "
                f'({t//(3600):02d}:{(t%3600)//60:02d}:{t%60:02d})')
        self.save(save_path)
        clear_output(wait=True)

      # Save history
      history['Gx'].append(running_loss_gx/steps_per_epoch)
      history['Gz'].append(running_loss_gz/steps_per_epoch)
      history['Dx'].append(running_loss_dx/steps_per_epoch)
      history['Dz'].append(running_loss_dz/steps_per_epoch)
      history['Rx'].append(running_loss_rx/steps_per_epoch)
      history['Rz'].append(running_loss_rz/steps_per_epoch)
      print(f'Gx: {history["Gx"][-1]}, of {len(history["Gx"])}')
      print(f'Gz: {history["Gz"][-1]}, of {len(history["Gz"])}')
      print(f'Dx: {history["Dx"][-1]}, of {len(history["Dx"])}')
      print(f'Dz: {history["Dz"][-1]}, of {len(history["Dz"])}')
      print(f'Rx: {history["Rx"][-1]}, of {len(history["Rx"])}')
      print(f'Rz: {history["Rz"][-1]}, of {len(history["Rz"])}')

      if show_predictions:
        print(f'\nImages generated in epoch: {epoch+1}')
        self._generate(save_path=save_path, epoch=epoch) 
        self.save(save_path)

    print('\nFinal Images')
    self._generate(save_path=save_path, epoch=epochs+1)
    self.save(save_path)
    return history

  def load(self, path):
    self.G = keras.models.load_model(os.path.join(path, 'aegan_Generator.keras'))
    self.E = keras.models.load_model(os.path.join(path, 'aegan_Encoder.keras'))
    self.D_x = keras.models.load_model(os.path.join(path, 'aegan_DiscriminatorImage.keras'))
    self.D_z = keras.models.load_model(os.path.join(path, 'aegan_DiscriminatorLatent.keras'))
    self.combined = keras.models.load_model(os.path.join(path, 'aegan_Combined.keras'))