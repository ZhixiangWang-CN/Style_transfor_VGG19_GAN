from keras import backend as K

def feature_reconstruction_loss(base, output):
	"""
	Feature reconstruction loss function. Encourages the 
	output img to be perceptually similar to the base image.
	"""
	return K.sum(K.square(output - base))

def gram_matrix(x):
	"""
	Computes the outer-product of the input tensor x.

	Input
	-----
	- x: input tensor of shape (C x H x W)

	Returns
	-------
	- x . x^T

	Note that this can be computed efficiently if x is reshaped
	as a tensor of shape (C x H*W).
	"""
	# assert K.ndim(x) == 3
	if K.image_dim_ordering() == 'th':
		features = K.batch_flatten(x)
	else:
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	return K.dot(features, K.transpose(features))

def style_reconstruction_loss(base,base2, output, img_nrows, img_ncols):
	"""
	Style reconstruction loss. Encourage the output img 
	to have same stylistic features as style image. Does not
	preserve spatial structure however.
	"""
	####这里就是修改style特征的损失函数的地方，base是style1，base2，是style2.最后这个函数就是计算输出的和style1,2的区别
	H, W, C = img_nrows, img_ncols, 3
	gram_base = gram_matrix(base)
	gram_output = gram_matrix(output)
	gram_base2 = gram_matrix(base2)
	factor = 1.0 / float((2*C*H*W)**2)
	out1 = factor * K.sum(K.square(gram_output - gram_base))
	out2 = factor * K.sum(K.square(gram_output - gram_base2))
	out = out1+out2
	return out

def total_variation_loss(x, img_nrows, img_ncols):
	"""
	Total variational loss. Encourages spatial smoothness 
	in the output image.
	"""
	H, W = img_nrows, img_ncols
	if K.image_dim_ordering() == 'th':
		a = K.square(x[:, :, :H-1, :W-1] - x[:, :, 1:, :W-1])
		b = K.square(x[:, :, :H-1, :W-1] - x[:, :, :H-1, 1:])
	else:	
		a = K.square(x[:, :H-1, :W-1, :] - x[:, 1:, :W-1, :])
		b = K.square(x[:, :H-1, :W-1, :] - x[:, :H-1, 1:, :])

	return K.sum(K.pow(a + b, 1.25))
