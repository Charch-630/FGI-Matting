import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_Attn(nn.Module):

	def __init__(self, in_dim, with_attention=False):
		super (Self_Attn, self).__init__ ()
		self.chanel_in = in_dim
		# self.activation = activation
		self.with_attention = with_attention

		self.query_conv = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv = nn.Conv2d (in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros (1))

		self.softmax = nn.Softmax(dim=-1)  

	def forward(self, x):
		"""
			inputs :
				x : input feature maps( B X C X H X W)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize, C, width, height = x.size ()
		proj_query = self.query_conv (x).view (m_batchsize, -1, width * height).permute (0, 2, 1)  # B X N X C
		proj_key = self.key_conv (x).view (m_batchsize, -1, width * height)  # B X C x (*W*H)
		energy = torch.bmm (proj_query, proj_key)  # transpose check
		attention = self.softmax (energy)  # BX (N) X (N)
		proj_value = self.value_conv (x).view (m_batchsize, -1, width * height)  # B X C X N

		out = torch.bmm (proj_value, attention.permute (0, 2, 1))
		out = out.view (m_batchsize, C, width, height)#B X C X H X W

		out = self.gamma * out + x

		if self.with_attention:
			return out, attention
		else:
			return out



class Self_Attn_trimap(nn.Module):

	def __init__(self, in_dim, with_attention=False):
		super (Self_Attn_trimap, self).__init__ ()
		self.chanel_in = in_dim
		# self.activation = activation
		self.with_attention = with_attention

		self.query_conv_fg = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv_fg = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv_fg = nn.Conv2d (in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma_fg = nn.Parameter(torch.zeros (1))
		# self.sigmoid_fg = nn.Sigmoid()

		self.query_conv_bg = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv_bg = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv_bg = nn.Conv2d (in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma_bg = nn.Parameter(torch.zeros (1))
		# self.sigmoid_bg = nn.Sigmoid()

		self.query_conv_transition = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv_transition = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv_transition = nn.Conv2d (in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma_transition = nn.Parameter(torch.zeros (1))
		# self.sigmoid_transition = nn.Sigmoid()

		


	def fg_attention(self, x, trimap_fg):
		m_batchsize, C, width, height = x.size()
		x = x * trimap_fg
		proj_query = self.query_conv_fg(x).view (m_batchsize, -1, width * height).permute (0, 2, 1)  # B X N X C
		proj_key = self.key_conv_fg(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
		energy = torch.bmm(proj_query, proj_key)  # transpose check
		attention = torch.sigmoid(energy)  # BX (N) X (N)
		proj_value = self.value_conv_fg(x).view (m_batchsize, -1, width * height)  # B X C X N

		out = torch.bmm (proj_value, attention.permute (0, 2, 1))
		out = out.view (m_batchsize, C, width, height)#B X C X H X W

		return out

	def transition_attention(self, x, trimap_transition):
		m_batchsize, C, width, height = x.size()
		x = x * trimap_transition
		proj_query = self.query_conv_transition(x).view (m_batchsize, -1, width * height).permute (0, 2, 1)  # B X N X C
		proj_key = self.key_conv_transition(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
		energy = torch.bmm(proj_query, proj_key)  # transpose check
		attention = torch.sigmoid(energy)  # BX (N) X (N)
		proj_value = self.value_conv_transition(x).view (m_batchsize, -1, width * height)  # B X C X N

		out = torch.bmm (proj_value, attention.permute (0, 2, 1))
		out = out.view (m_batchsize, C, width, height)#B X C X H X W

		return out

	def bg_attention(self, x, trimap_bg):
		m_batchsize, C, width, height = x.size()
		x = x * trimap_bg
		proj_query = self.query_conv_bg(x).view (m_batchsize, -1, width * height).permute (0, 2, 1)  # B X N X C
		proj_key = self.key_conv_bg(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
		energy = torch.bmm(proj_query, proj_key)  # transpose check
		attention = torch.sigmoid(energy)  # BX (N) X (N)
		proj_value = self.value_conv_bg(x).view (m_batchsize, -1, width * height)  # B X C X N

		out = torch.bmm (proj_value, attention.permute (0, 2, 1))
		out = out.view (m_batchsize, C, width, height)#B X C X H X W

		return out



	def forward(self, x, trimap):
		"""
			inputs :
				x : input feature maps( B X C X H X W)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
			trimap  :
				通道  0：背景 1：过渡 2：前景
		"""
		N,C,H,W = x.size()
		x = F.interpolate(x, (H//4,W//4), mode="bilinear", align_corners=False)
		trimap = F.interpolate(trimap, (H//4,W//4), mode="bilinear", align_corners=False)

		trimap_bg = trimap[:,0:1,:,:]
		trimap_transition = trimap[:,1:2,:,:]
		trimap_fg = trimap[:,2:3,:,:]


		out_fg = self.fg_attention(x, trimap_fg)
		out_transition = self.transition_attention(x, trimap_transition)
		out_bg = self.bg_attention(x, trimap_bg)


		out = self.gamma_fg * out_fg+self.gamma_transition * out_transition + self.gamma_bg * out_bg + x

		out = F.interpolate(out, (H,W), mode="bilinear", align_corners=False)

		if self.with_attention:
			return out, attention
		else:
			return out


if __name__=="__main__":
	import time
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]='0'

	# net = Self_Attn(32).cuda()
	# for i in range(50):
	# 	inp1 = torch.zeros([4,32,64,64]).float().cuda()
	# 	t1 = time.time()
	# 	oup = net(inp1)
	# 	print("time:",time.time()-t1)
	# print(oup.size())


	net = Self_Attn_trimap(64).eval().cuda()
	with torch.no_grad():
		for i in range(50):
			inp1 = torch.rand([10,64,128,128]).float().cuda()
			trimap = torch.rand([10,3,128,128]).float().cuda()
			t1 = time.time()
			oup = net(inp1, trimap)
			print("time:",time.time()-t1)

	print(oup.size())

