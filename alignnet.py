import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import math

prng_key = jax.random.key(42)   # pseudorandom number generator

print('---------------------------------------')
print(' Check the JAX devices')
print('---------------------------------------')

print('jax devices')
print(jax.devices())

print('jax backend')
print(jax.default_backend())

print('---------------------------------------')
print(' Load CIFAR10 dataset')
print('---------------------------------------')

# Training - includes augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # flip left-right 50% of the time
    transforms.RandomCrop(32, padding=4),       # pad by 4 then random crop back to 32x32
    transforms.ColorJitter(brightness=0.2,      # slight color variation
                           contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
                         (0.2470, 0.2435, 0.2616))    # CIFAR-10 std
])

# Test - no augmentation, just normalize
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])


# Download and load dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

# Class names for reference
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
	   'dog', 'frog', 'horse', 'ship', 'truck']
n_class = 10

print('---------------------------------------')
print(' Define AlignNet vision model')
print('---------------------------------------')

#
# input:
#  x    B H W C1
#  A    C1 C2
# output:
#  res  B H W C2
#
def Linear(x,A):
	B,H,W,C1 = x.shape
	C1,C2    = A.shape
	x = x.reshape((B,H,W,1,C1))
	A = A.reshape((1,1,1,C1,C2))
	res = jnp.matmul(x,A)        # B H W 1 C2
	res = res.reshape((B,H,W,C2))
	return res

#
# input:
#   x     B H W C
# output:
#   res   B H W 2*C
#
def SplitLU(x):
	pos = jnp.where(x > 0.0, x, 0.0)  # (B,H,W,C)
	neg = jnp.where(x < 0.0, x, 0.0)  # (B,H,W,C)
	#print('===')
	#print('SplitLU')
	#print('x.shape', x.shape)
	#print('pos.shape', pos.shape)
	#print('neg.shape', neg.shape)
	res = jnp.concatenate((pos, neg), axis=3)   # (B,H,W,2*C)
	return res

#----------------
# Average pool the channels
#----------------
def AvgPool(x):
	B,H,W,C = x.shape
	outH = H//2
	outW = W//2
	x = x.reshape((B,outH,2,outW,2,C))
	x = jnp.mean(x, axis=(2,4))        # B H//2 W//2 C
	return x

#----------------
#  BatchNorm
#  input:
#    C         channels for batchnorm
#    falloff   exponential moving average coefficient
#----------------
def BatchNorm(C, falloff=0.99):
	
	#---
	# output:
	#   con   (mu,sig)  non-differential constraints
	#---
	def init():
		con = {}    # constraints
		con['mu']  = jnp.zeros((C,), dtype='float32')
		con['sig'] = jnp.ones((C,),  dtype='float32')
		return con
		
	#---
	# input:
	#   con   constraints  (mu,sig)
	#   x     input (B,H,W,C)
	# output:
	#   res   output (B,H,W,C)
	#   cup   constraint updates (mu,sig)
	#---
	def forward(con, x):
		
		# Apply batch-norm
		mu  = con['mu'].reshape((1,1,1,C))     # (1,1,1,C)
		sig = con['sig'].reshape((1,1,1,C))    # (1,1,1,C)
		inv_sig = 1.0 / sig                    # (1,1,1,C)
		diff = x-mu                            # (B,H,W,C)
		res = diff * inv_sig                   # (B,H,W,C)
		
		# Constraint updating     (jit will remove during test)
		batch_mu  = jnp.mean(x,         axis=(0,1,2))  # (C,)
		diff2 = x-batch_mu.reshape((1,1,1,C))
		batch_sig = jnp.mean(diff2*diff2, axis=(0,1,2))  # (C,)
		batch_sig = jnp.sqrt(batch_sig)                # (C,)
		cup = {}
		#print('mu', mu)
		#print('batch_mu', batch_mu)
		#input('enter')
		#print('sig', sig)
		#print('batch_sig', batch_sig)
		#input('enter')
		cup['mu']  = con['mu']*falloff + batch_mu*(1.0-falloff)
		cup['sig'] = con['sig']*falloff + batch_sig*(1.0-falloff)	
		return x,cup
	
	return init,forward
	

#----------------
#  SplitConvBlock
#  input:
#    C0   input channels
#    C1   hidden channels
#    C3   output channels
#    H,W  height and width
#----------------
def SplitConvBlock(key, C0, C1, C2, H, W):
	bn1_init,bn1_forward = BatchNorm(C1)
	bn2_init,bn2_forward = BatchNorm(C1)
	bn3_init,bn3_forward = BatchNorm(C2)
	keys = jax.random.split(key, num=3)
	
	#---
	# output:
	#   par   differential parameters
	#   con   non-differential constraints
	#---
	def init():
		global seed
		par = {}   # parameters
		con = {}   # constraints
		par['conv1'] = jax.random.normal(keys[0], (2*C0,C1))   * math.sqrt(2.0 / (2*C0 + C1))
		par['conv1_bias'] = jnp.zeros((C1,))
		con['bn1']   = bn1_init()
		par['conv2'] = jax.random.normal(keys[1], (3,3,C1,C1)) * math.sqrt(2.0 / (3*3*C1 + C1))
		par['conv2_bias'] = jnp.zeros((C1,))
		con['bn2']   = bn2_init()
		par['conv3'] = jax.random.normal(keys[2], (2*C1,C2))   * math.sqrt(2.0 / (2*C1 + C2))
		par['conv3_bias'] = jnp.zeros((C2,))
		con['bn3']   = bn3_init()
		return par,con
	
	#---
	#  input:
	#      par,con            parameters and constraints
	#      x   (B,H,W,C0)
	#  output:
	#      x   (B,H,W,C2)
	#      cup                constraint updates
	#---
	def forward(par,con,x):
		cup = {}
		x = SplitLU(x)                                   # x: B H W 2*C0
		x = Linear(x, par['conv1'])                  # x: B H W C1
		x,cup['bn1'] = bn1_forward(con['bn1'], x)        # x: B H W C1    cup:  mu sig
		x = x + par['conv1_bias'].reshape((1,1,1,C1))
	
		#print('==========')
		#print('conv_general')
		#print('x.shape', x.shape)
		#print('par[conv2].shape', par['conv2'].shape)
		x = jax.lax.conv_general_dilated(                # x: B H W C1
				x, par['conv2'], window_strides=(1, 1),
				padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
		x,cup['bn2'] = bn2_forward(con['bn2'], x)        # x: B H W C1
		x = x + par['conv2_bias'].reshape((1,1,1,C1))
		
		x = SplitLU(x)                                   # x: B H W 2*C1
		x = Linear(x, par['conv3'])                  # x: B H W C2
		x,cup['bn3'] = bn3_forward(con['bn3'], x)
		x = x + par['conv3_bias'].reshape((1,1,1,C2))
		
		return x,cup

	return init,forward



def AlignNet(key, inH=32, inW=32, inChan=3, C_base=10, n_layers=0, n_class=10):
	
	#--
	# Shapes throught network depth
	#--
	H = inH
	W = inW
	C = C_base
	layer_shapes = [(inH,inW,inChan), (H,W,C)]   # input projection
	stem_bn_init,stem_bn_forward = BatchNorm(C_base)
	keys = jax.random.split(key, num=(n_layers+1))

	#--
	# Layer configuration
	#--
	conv_block_init    = []
	conv_block_forward = []
	for i in range(n_layers):
		C0 = C
		C1 = C//2
		C2 = 2*C
		init_,forward_ = SplitConvBlock(keys[i+1], C0, C1, C2, H, W)
		conv_block_init.append   ( init_ )
		conv_block_forward.append( forward_ )
		H //= 2
		W //= 2
		C *= 2
		layer_shapes.append( (H,W,C) )        # After pooling
	
	#---
	# output:
	#   par   differential parameters
	#   con   non-differential constraints
	#---
	def init():
		global seed
		par = {}
		con = {}
		par['stem'] = jax.random.normal(keys[0], (inChan,C_base))  *  math.sqrt(2.0 / (inChan + C_base))
		con['stem_bn'] = stem_bn_init()
		
		par['conv_block'] = []
		con['conv_block'] = []
		for i in range(n_layers):
			par_,con_ = conv_block_init[i]()
			par['conv_block'].append( par_ )
			con['conv_block'].append( con_ )
		return par,con
	
	#---
	#  input:
	#      par,con            parameters and constraints
	#      x   (B,H,W,C)
	#  output:
	#      featvec   [n_layers]  each of format B,H,W,C
	#      cup       constraint updates
	#---
	def forward(par,con,x):
		cup = {'conv_blocks':[]}
		
		featvec = []
		featvec.append(x)
		
		#print('=========')
		#print('10 x.shape', x.shape)
		#print('10 par[stem].shape', par['stem'].shape)
		
		# Apply stem projection
		x = Linear(x,par['stem'])

		#print('20 x.shape', x.shape)

		x,cup['stem_bn'] = stem_bn_forward(con['stem_bn'],x)
		featvec.append(x)

		#print('30 x.shape', x.shape)
		
		# Apply each layer
		cup['conv_block'] = []
		for i in range(n_layers):
			#print('=======')
			#print('i', i)
			#print('par', type(par), 'con', type(con), 'x', type(x))
			#print('x.shape', x.shape)
			B,H,W,C = x.shape
			z,cup_ = conv_block_forward[i](     # B H W 2*C
				par['conv_block'][i], con['conv_block'][i], x)
			cup['conv_block'].append( cup_ )
			
			# Apply fancy skip connection
			z = z.reshape((B,H,W,C,2))
			x = x.reshape((B,H,W,C,1))
			x = x + z                     # B H W C 2
			C = 2*C
			x = x.reshape((B,H,W,C))      # B H W C
			
			# Apply pooling
			H = H//2
			W = W//2
			x = AvgPool(x)                # B H W C
			featvec.append(x)
			
		return featvec,cup	

	return init,forward

print('---------------------------------------')
print(' Define the Training logic')
print('---------------------------------------')

#-------------
# Build the network
#-------------
model_init,model_forward = AlignNet(prng_key, C_base=5, n_layers=5)
model_par,model_con = model_init()

#print(model_par)
#input('model_par  enter')

#print(model_con)
#input('model_con  enter')

def log_sigmoid_loss(yhat,y):
	B,n_class = yhat.shape
	loss = -y * jax.nn.log_sigmoid(yhat)
	loss = jnp.sum(loss) / n_class
	return loss

#
# Input:
#     x   B H W C
# Output:
#     zhat   B n_class
#
def calc_zhat(x,n_class):
	B,H,W,C = x.shape
	K = C // n_class
	x = x.reshape((B,H,W,n_class,K))
	zhat = jnp.mean(x, axis=(1,2,4))
	return zhat

#----
# train_forward
#
# input:
#   par    model parameters
#   con    model constraints
#   X      B H W chan
#   label  B
# output:   loss,(yhat,layer_loss,cup)
#   loss          1,               overall loss
#   yhat                           penultimate layer logits
#   layer_loss    train_stage      loss per training stage
#   cup                            constraint updates
#----
def train_forward(par,con,X,label):
	Y = jax.nn.one_hot(label, n_class)
	

	featvec,cup = model_forward(par,con,X)
	
	#len_featvec = min(train_stage+3,len(featvec))
	len_featvec = len(featvec)
	
	loss = 0.0
	layer_loss = []
	for i in range(2,len_featvec):
		yhat = calc_zhat(featvec[i],n_class)
		loss_ = log_sigmoid_loss(yhat,Y)
		layer_loss.append( loss_ )
		loss = loss_ + loss
	
	return loss,(yhat,layer_loss,cup)

#----
# train_forward
#
# input:
#   par    model parameters
#   con    model constraints
#   X      B H W chan
#   label  B
#   lr     learning rate
# output:   loss,(yhat,layer_loss,cup)
#   pup           updated parameters
#   cup           updated constraints
#   loss          1,               overall loss
#   yhat                           penultimate layer logits
#   layer_loss    train_stage      loss per training stage
#----
def train_batch(par,con,X,label,lr):
	
	#---
	# Run the forward and backward training step
	#---
	(loss, aux), grad = jax.value_and_grad(train_forward, has_aux=True)(par,con,X,label)
	yhat,layer_loss,cup = aux
	
	
	#---
	# Update the gradient using basic SGD
	#---
	pup = jax.tree.map(lambda p,g: p - lr*g, par, grad)
	
	#print('grad', grad)
	#input('enter')
	
	#print('par', par)
	#input('enter')	
	
	#print('pup', pup)
	#input('enter')
	
	#---
	# Return the updated parameters and constriants
	#---
	return pup,cup,loss,yhat,layer_loss
	



print('---------------------------------------')
print(' Training loop')
print('---------------------------------------')

train_lr = 0.1
train_lr_falloff = 0.97 #0.986

print('---------------')
print(' Compile the function')
print('---------------')
train_batch_jit = jax.jit(train_batch)

for epoch in range(50):
	
	
	#print('stem_bn', model_con['stem_bn'])
	#print('bn1', model_con['conv_block'][0]['bn1'])
	#print('bn2', model_con['conv_block'][0]['bn2'])
	#print('bn3', model_con['conv_block'][0]['bn3'])
	
	epoch_loss = 0.0
	
	# Example: iterate and convert to JAX arrays
	for batchno, (images, labels) in enumerate(train_loader):
		images = np.ascontiguousarray(images.numpy().transpose(0, 2, 3, 1))
		images = jnp.array(images)          # (B,sY,sX,C)
		labels = jnp.array(labels.numpy())  # (B,)

		#print('images[0]', images[0])

		#B,H,W,C = images.shape
		#for b in range(B):
		#	for h in range(H):
		#		for w in range(W):
		#			print('(%.2f %.2f %.2f)' % (float(images[b,h,w,0]), float(images[b,h,w,1]), float(images[b,h,w,2])), end=' ')
		#		print('')
		#	print('')
		#	input('enter')

		#print ('images', images.shape, images.dtype)
		#print ('labels', labels.shape, labels.dtype)


		# Run the training batch
		#with jax.disable_jit():
		pup,cup,loss,yhat,layer_loss = train_batch_jit(model_par,model_con,images,labels,train_lr)
		
		# Update the epoch layer loss
		if batchno==0:
			epoch_layer_loss = layer_loss
		else:
			for i in range(len(layer_loss)):
				epoch_layer_loss[i] = epoch_layer_loss[i] + layer_loss[i]
		
		# Update the weights and constraints
		model_par = pup
		model_con = cup
		
		epoch_loss = loss + epoch_loss

	# Divide the layer loss
	epoch_loss /= batchno
	for i in range(len(epoch_layer_loss)):
		epoch_layer_loss[i] /= batchno

	# Print the epoch and loss
	print('epoch %d lr %.3f loss %.7f  layer_loss' % (epoch, train_lr, float(epoch_loss)), end=' ')
	for i in range(len(epoch_layer_loss)):
		print('%.3f' % epoch_layer_loss[i], end=' ')
	print('')
	
	train_lr *= train_lr_falloff

	
print('Done!')	
