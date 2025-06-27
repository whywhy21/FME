"""Pytorch Mechine Learning Model"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set random seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Create FNO1d
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.dim1 = dim1
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels, self.modes1), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, modes on t), (in_channel, out_channel, modes on t) -> (batch, out_channel, modes on t)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1=None):

        if dim1 is not None:
            self.dim1 = dim1

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1 , norm = 'forward')
        return x

class SpectralConv1d_Freq(nn.Module):
    def __init__(self, in_channels, out_channels, dim1,fr=100, low_freq = 0,high_freq = 5):
        super(SpectralConv1d_Freq, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_start = int(low_freq*dim1/fr)
        self.modes_end = int(high_freq*dim1/fr)  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.dim1 = dim1
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels, self.modes_end-self.modes_start), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, modes on t), (in_channel, out_channel, modes on t) -> (batch, out_channel, modes on t)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1=None):
        #print(self.modes_start,self.modes_end)
        if dim1 is not None:
            self.dim1 = dim1

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, self.modes_start:self.modes_end] = self.compl_mul1d(x_ft[:, :, self.modes_start:self.modes_end], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1 , norm = 'forward')
        return x

class pointwise_op_1D(nn.Module):
    """
    All variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1):
        super(pointwise_op_1D,self).__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        if dim1 is None:
            dim1 = self.dim1
        #print('conv2d input',x.shape)
        x_out = self.conv(x)
        #print('conv2d output',x_out.shape)
        x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True)#, antialias= True)
        #print('interpolate output',x_out.shape)
        return x_out

class FNO1D(nn.Module):
    """
    Normalize = if true performs InstanceNorm1d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim, modes1, dim1, Normalize = True, Non_Lin = True):
        super(FNO1D,self).__init__()
        self.conv = SpectralConv1d(in_codim, out_codim, int(dim1), int(modes1))
        self.w = pointwise_op_1D(in_codim, out_codim, int(dim1))
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)

    def forward(self,x):#, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        x1_out = self.conv(x)#,dim1)
        x2_out = self.w(x)#,dim1)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class FNO1D_fr(nn.Module):
    """
    Normalize = if true performs InstanceNorm1d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim, dim1 ,fr,low_freq,high_freq, Normalize = True, Non_Lin = True):
        super(FNO1D_fr,self).__init__()
        self.conv = SpectralConv1d_Freq(in_codim, out_codim, int(dim1),int(fr),low_freq,high_freq)
        self.w = pointwise_op_1D(in_codim, out_codim, int(dim1))
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)

    def forward(self,x):#, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        x1_out = self.conv(x)#,dim1)
        x2_out = self.w(x)#,dim1)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class FME_fr(nn.Module):
  def __init__(self, num_classes = 1, modes = 24, width = 3, n_stations = 10):
    super(FME_fr, self).__init__()
    self.modes1 = modes
    self.width  = width
    self.padding = 50
    self.in_len = 2000
    self.hid_len = 750
    self.out_len = 200
    self.fr = 100
    self.low_freq = 0
    self.high_freq = 5
    self.n_stations = n_stations

    self.FNO_layers = nn.ModuleList([
            nn.Sequential(
                FNO1D_fr(self.width, self.width, self.in_len + self.padding * 2, self.fr,self.low_freq,self.high_freq),
                FNO1D(self.width, self.width * 4, self.modes1 // 2, self.hid_len),
                FNO1D(self.width * 4, 1, self.modes1 // 3, self.out_len)
            ) for _ in range(n_stations)
        ])

    self.lstm = nn.LSTM(200,100, bidirectional=True, batch_first=True) # 30s 187,100 20s 125,100
    self.disfc = nn.Linear(1, 1)
    self.fc1 = nn.Linear(200*10, num_classes)

  def forward(self, x):
    x1 = x[:, :, :, :2000]  # batch, channel, n_stations, t
    x2 = x[:, 2:3, :, 2000:2001]  # batch, 1, n_stations,1
    x2 = self.disfc(x2)

    # Apply FNO layers on each station
    x1 = torch.unbind(x1, dim=2)
    x1 = torch.stack([layer(x1[i]) for i, layer in enumerate(self.FNO_layers)], dim=2)  

    # Multiply outputs with distance feature
    x = x1 * x2  # (batch, 1, n_stations, out_len)
    x = torch.squeeze(x, dim=1)  # (batch, n_stations, out_len)

    # Pass through LSTM
    x, (_, _) = self.lstm(x)

    # Flatten and pass through FC layer
    x = x.reshape(x.shape[0], -1)
    #x = torch.flatten(x)
    x = self.fc1(x)

    return x

class FMEModel():
    def __init__(self,num_classes=1, device=None , model = 'Model1',loss = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model == 'Model1':
           self.model = FME_fr(num_classes=num_classes).to(self.device)
        else:
            print(f"didn't find model:{model} ,please check")
            raise ValueError
        self.loss = loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def load(self, path, optim=False):
        loader = torch.load(path)
        self.model.load_state_dict(loader['state_dict'])
        if optim:
            self.optimizer.load_state_dict(loader['optimizer'])
        print('=> Loading checkpoint')
    
    def save(self, path):
        state = {'state_dict' : self.model.state_dict(), 
                'optimizer':self.optimizer.state_dict()}
        torch.save(state, path)
        print('=> save checkpoint')

    def predict(self, x, y):
        self.model.eval()
        with torch.no_grad():
            py = self.model(x)
            try:
                loss = self.loss_funtion(py,y)
            except:
                loss = None
        self.model.train()
        return py,loss
    
    def train(self, x, y):
        py = self.model(x)
        loss = self.loss_funtion(py,y)
        return py,loss
    
    def loss_funtion(self,input, target, size_average=True):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
        if self.loss == 'MSE':
            muti_time = torch.exp(target*0.5)/torch.exp(torch.tensor(3))
            L = muti_time * (input - target) ** 2
            return torch.mean(L) if size_average else torch.sum(L)
        if self.loss == 'CE':
           loss = torch.nn.functional.cross_entropy(input, target)
           return loss

