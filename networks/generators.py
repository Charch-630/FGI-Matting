import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn

from   utils import CONFIG
from   networks import encoders, decoders


class Generator(nn.Module):
    def __init__(self, encoder, decoder):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, trimap):
        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha, info_dict = self.decoder(embedding, mid_fea, trimap)

        return alpha, info_dict


def get_generator(encoder, decoder):
    generator = Generator(encoder=encoder, decoder=decoder)
    return generator


if __name__=="__main__":
    import time
    generator = get_generator(encoder="res_shortcut_encoder_29_spatial_attn", decoder="res_shortcut_decoder_22_spatial_attn").eval()
    generator = generator.cuda()
    time_all = 0
    with torch.no_grad():
        for i in range(50):
            inp1 = torch.rand([1,3,1024,1024]).float().cuda()
            inp2 = torch.rand([1,1,1024,1024]).float().cuda()
            t1 = time.time()
            oup = generator(inp1, inp2)

            time_p = time.time()-t1
            time_all+= time_p
            print("time:", time_p)

    print(oup[0].size())
    print("avg time:", time_all/50)
    