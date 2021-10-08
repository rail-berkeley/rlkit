import os
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
from torchvision import transforms


class DMPIntegrator:
    def __init__(self, rbf="gaussian", only_g=False, az=False):
        a = 1
        self.rbf = rbf
        self.only_g = only_g
        self.az = az
        # self.x = 1

    def forward(
        self,
        inputs,
        parameters,
        param_gradients,
        scaling,
        y0,
        dy0,
        goal=None,
        w=None,
        vel=False,
    ):
        dim = int(parameters[0].item())
        k = dim
        N = int(parameters[1].item())
        division = k * (N + 2)
        inputs_np = inputs
        if goal is not None:
            goal = goal
            w = w
        else:
            w = inputs_np[:, dim : dim * (N + 1)]
            goal = inputs_np[:, :dim]

        if self.az:
            alpha_z = inputs[:, -1]
            t = y0.shape[0] // inputs.shape[0]
            alpha_z = alpha_z.repeat(t, 1).transpose(0, 1).reshape(inputs.shape[0], -1)
            alpha_z = alpha_z.contiguous().view(
                alpha_z.shape[0] * alpha_z.shape[1],
            )

        w = w.reshape(-1, N)

        if self.only_g:
            w = torch.zeros_like(w)
        if vel:
            dy0 = torch.zeros_like(y0)

        goal = goal.contiguous().view(
            goal.shape[0] * goal.shape[1],
        )

        if self.az:
            X, dX, ddX = integrate(
                parameters, w, y0, dy0, goal, 1, rbf=self.rbf, az=True, alpha_z=alpha_z
            )
        else:
            X, dX, ddX = integrate(parameters, w, y0, dy0, goal, 1, rbf=self.rbf)
        return inputs.new(X), inputs.new(dX), inputs.new(ddX)

    def forward_not_int(
        self,
        inputs,
        parameters,
        param_gradients,
        scaling,
        y0,
        dy0,
        goal=None,
        w=None,
        vel=False,
    ):
        dim = int(parameters[0].item())
        k = dim
        N = int(parameters[1].item())
        division = k * (N + 2)
        inputs_np = inputs
        if goal is not None:
            goal = goal
            w = w
        else:
            w = inputs_np[:, dim : dim * (N + 1)]
            goal = inputs_np[:, :dim]
        w = w.reshape(-1, N)
        if vel:
            dy0 = torch.zeros_like(y0)
        goal = goal.contiguous().view(
            goal.shape[0] * goal.shape[1],
        )
        return parameters, w, y0, dy0, goal, 1

    def first_step(self, w, parameters, scaling, y0, dy0, l, tau=1):
        data = parameters
        y = y0
        self.y0 = y0
        z = dy0 * tau
        self.x = 1
        self.N = int(data[1].item())
        self.dt = data[3].item()
        self.a_x = data[4].item()
        self.a_z = data[5].item()
        self.b_z = self.a_z / 4
        self.h = data[(6 + self.N) : (6 + self.N * 2)]
        self.c = data[6 : (6 + self.N)]
        self.num_steps = int(data[2].item()) - 1
        self.i = 0
        self.w = w.reshape(-1, self.N)
        self.tau = tau
        self.l = l

    def step(self, g, y, dy):
        g = g.reshape(-1, 1)[:, 0]
        z = dy * self.tau
        dt = self.dt
        for _ in range(self.l):
            dx = (-self.a_x * self.x) / self.tau
            self.x = self.x + dx * dt
            psi = torch.exp(-self.h * torch.pow((self.x - self.c), 2))
            fx = torch.mv(self.w, psi) * self.x * (g - self.y0) / torch.sum(psi)
            dz = self.a_z * (self.b_z * (g - y) - z) + fx
            dy = z
            dz = dz / self.tau
            dy = dy / self.tau
            y = y + dy * dt
            z = z + dz * dt
        self.i += 1
        return y, dy, dz


def integrate(data, w, y0, dy0, goal, tau, rbf="gaussian", az=False, alpha_z=None):
    y = y0
    z = dy0 * tau
    x = 1
    if w.is_cuda:
        Y = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
        dY = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
        ddY = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
    else:
        Y = torch.zeros((w.shape[0], int(data[2].item())))
        dY = torch.zeros((w.shape[0], int(data[2].item())))
        ddY = torch.zeros((w.shape[0], int(data[2].item())))
    Y[:, 0] = y
    dY[:, 0] = dy0
    ddY[:, 0] = z
    N = int(data[1].item())
    dt = data[3].item()
    a_x = data[4].item()
    a_z = data[5].item()
    if az:
        a_z = alpha_z
        a_z = torch.clamp(a_z, 0.5, 30)
    b_z = a_z / 4
    h = data[(6 + N) : (6 + N * 2)]
    c = data[6 : (6 + N)]
    for i in range(0, int(data[2].item()) - 1):
        dx = (-a_x * x) / tau
        x = x + dx * dt
        eps = torch.pow((x - c), 2)
        if rbf == "gaussian":
            psi = torch.exp(-h * eps)
        if rbf == "multiquadric":
            psi = torch.sqrt(1 + h * eps)
        if rbf == "inverse_quadric":
            psi = 1 / (1 + h * eps)
        if rbf == "inverse_multiquadric":
            psi = 1 / torch.sqrt(1 + h * eps)
        if rbf == "linear":
            psi = h * eps
        # psi = torch.exp(-h * torch.pow((x - c), 2))
        fx = torch.mv(w, psi) * x * (goal - y0) / torch.sum(psi)
        dz = a_z * (b_z * (goal - y) - z) + fx
        dy = z
        dz = dz / tau
        dy = dy / tau
        y = y + dy * dt
        z = z + dz * dt
        Y[:, i + 1] = y
        dY[:, i + 1] = dy
        ddY[:, i + 1] = dz
    return Y, dY, ddY


class DMPParameters:
    def __init__(self, N, tau, dt, Dof, scale, a_z=25):
        self.a_z = a_z
        self.a_x = 1
        self.N = N
        c = np.exp(-self.a_x * np.linspace(0, 1, self.N))
        sigma2 = np.ones(self.N) * self.N ** 1.5 / c / self.a_x
        self.c = torch.from_numpy(c).float()
        self.sigma2 = torch.from_numpy(sigma2).float()
        self.tau = tau
        self.dt = dt
        self.time_steps = int(np.round(self.tau / self.dt)) + 1
        self.y0 = [0]
        self.dy0 = np.zeros(Dof)
        self.Dof = Dof
        self.Y = torch.zeros((self.time_steps))
        grad = torch.zeros((self.time_steps, 2))
        self.data = {
            "time_steps": self.time_steps,
            "c": self.c,
            "sigma2": self.sigma2,
            "a_z": self.a_z,
            "a_x": self.a_x,
            "dt": self.dt,
            "Y": self.Y,
        }
        dmp_data = torch.tensor(
            [self.Dof, self.N, self.time_steps, self.dt, self.a_x, self.a_z]
        )
        data_tensor = torch.cat((dmp_data, self.c, self.sigma2), 0)
        data_tensor.dy0 = self.dy0
        data_tensor.tau = self.tau
        weights = torch.zeros((1, self.N))
        weights = torch.zeros((1, self.N))
        grad[:, 1], _, _ = integrate(data_tensor, weights, 0, 0, 1, self.tau)
        self.data_tensor = data_tensor
        self.grad_tensor = grad
        self.point_grads = torch.zeros(self.N * 2 + 4)
        self.X = np.zeros((self.time_steps, self.Dof))


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=220)
parser.add_argument("--T", type=int, default=249)
parser.add_argument("--l", type=int, default=1)
parser.add_argument("--expID", type=int, default=1)
parser.add_argument("--bs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--az", type=int, default=15)
parser.add_argument("--scale", type=int, default=5)
parser.add_argument("--axis", type=str, default="pos", help="pos|type|full")
parser.add_argument("--typ", type=int, default=0)
parser.add_argument("--freeze", type=int, default=0)
parser.add_argument("--num-epochs", type=int, default=5001)
parser.add_argument("--cuda", action="store_true", default=False)
args = parser.parse_args()


save_dir = "/home/sbahl2/research/dmp_rl/data/benchmark/zip/" + str(
    "{:05d}".format(args.expID)
)
T = args.T
algo = "demos"
num_epochs = args.num_epochs
os.makedirs(save_dir, exist_ok=True)
ims = np.load(
    "/home/sbahl2/research/dmp_rl/data/benchmark/new_pour/processed_ims.npy",
    allow_pickle=True,
).item()[algo]
trajs = np.load(
    "/home/sbahl2/research/dmp_rl/data/benchmark/new_pour/processed_trajs.npy",
    allow_pickle=True,
).item()[algo]


pos_train_inds = np.arange(15)
pos_test_inds = np.array([1, 2, 20, 30])

X_train = ims
Y_train = trajs
X_test = ims[pos_test_inds]
Y_test = trajs[pos_test_inds]

if len(X_train.shape) > 4:
    X_train, X_test, Y_train, Y_test = (
        np.concatenate(X_train, axis=0),
        np.concatenate(X_test, axis=0),
        np.concatenate(Y_train, axis=0),
        np.concatenate(Y_test, axis=0),
    )
X_train, X_test, Y_train, Y_test = (
    torch.Tensor(X_train),
    torch.Tensor(X_test),
    torch.Tensor(Y_train),
    torch.Tensor(Y_test),
)
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
Xs = []
for x in X_train:
    Xs.append(normalize(x.transpose(2, 0).transpose(1, 2)).unsqueeze(0))
X_train = torch.cat(Xs, dim=0)

Xs = []
for x in X_test:
    Xs.append(normalize(x.transpose(2, 0).transpose(1, 2)).unsqueeze(0))
X_test = torch.cat(Xs, dim=0)


class Net(nn.Module):
    def __init__(self, bias=None):
        super(Net, self).__init__()
        c1, a1 = nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c2, a2 = nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        m1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        )
        c3, a3 = nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c4, a4 = nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        self.vgg = nn.Sequential(c1, a1, c2, a2, m1, c3, a3, c4, a4)

        self.extra_convs = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        fc1, a1 = nn.Linear(256, 16), nn.ReLU(inplace=True)
        fc2 = nn.Linear(16, 3, bias=False)
        self.top = nn.Sequential(fc1, a1, fc2)
        bias = (
            np.zeros(3).astype(np.float32)
            if bias is None
            else np.array(bias).reshape(3)
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.from_numpy(bias).float(), requires_grad=True)
        )

    def forward(self, x):
        # vgg convs and 2D softmax
        x = self.vgg(x)
        x = self.extra_convs(x)
        B, C, H, W = x.shape
        x = F.softmax(x.view((B, C, H * W)), dim=2).view((B, C, H, W))

        # find expected keypoints
        h = torch.linspace(-1, 1, H).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 3)
        w = torch.linspace(-1, 1, W).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 2)
        x = torch.cat([torch.sum(a, 2) for a in (h, w)], 1)

        # regress final pose and add bias
        x = self.top(x) + self.bias
        return x

    def forward_traj(self, x):
        x = self.vgg(x)
        x = self.extra_convs(x)
        B, C, H, W = x.shape
        x = F.softmax(x.view((B, C, H * W)), dim=2).view((B, C, H, W))

        # find expected keypoints
        h = torch.linspace(-1, 1, H).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 3)
        w = torch.linspace(-1, 1, W).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 2)
        x = torch.cat([torch.sum(a, 2) for a in (h, w)], 1)
        return x


net = Net()
net.load_state_dict(torch.load("/home/sbahl2/smith_vgg/final.pt"))
net = net.eval()


class DMPNet(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        hidden_activation=torch.tanh,
        N=5,
        T=10,
        l=10,
        tau=1,
        goal_type="int_path",
        rbf="gaussian",
        num_layers=1,
        a_z=15,
        state_index=np.arange(7),
        freeze=False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.N = N
        self.l = l
        self.goal_type = goal_type
        self.vel_index = vel_index
        self.output_size = N * len(state_index) + len(state_index)
        output_size = self.output_size
        dt = tau / (T * self.l)
        self.T = T
        self.output_activation = torch.tanh
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, a_z=a_z)
        self.func = DMPIntegrator(rbf=rbf, only_g=False, az=False)
        self.register_buffer("DMPp", self.DMPparam.data_tensor)
        self.register_buffer("param_grad", self.DMPparam.grad_tensor)
        self.state_index = state_index
        self.output_dim = output_size
        self.hidden_activation = hidden_activation
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        if num_layers > 1:
            self.fc1 = init_(nn.Linear(40, hidden_size))
            self.fc2 = init_(nn.Linear(hidden_size, output_size // 2))
            self.layers = [self.fc1, self.fc2]
            self.fc_last = init_(nn.Linear(output_size // 2, output_size))
        else:
            self.fc_last = init_(nn.Linear(256, output_size))
            self.layers = []
        self.pt = net
        if freeze:
            for param in self.pt.parameters():
                param.requires_grad = False

    def forward(
        self, input, state=None, vel=None, return_preactivations=False, first_step=False
    ):
        h = self.pt.forward_traj(input)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        output = self.fc_last(h) * args.scale
        y0 = state[:, self.state_index].reshape(input.shape[0] * len(self.state_index))
        dy0 = torch.ones_like(y0) * 0.05
        # dy0 = velreshape(input.shape[0]*len(self.state_index))
        y, dy, ddy = self.func.forward(
            output, self.DMPp, self.param_grad, None, y0, dy0
        )
        y = y.view(input.shape[0], len(self.state_index), -1)
        y = y[:, :, :: self.l]
        return y.transpose(1, 2)


N = args.N
l = args.l
a_z = args.az
hidden_sizes = 100
state_index = np.arange(7)
vel_index = None
dmpn = DMPNet(
    N=N,
    l=l,
    T=T,
    a_z=a_z,
    hidden_size=hidden_sizes,
    state_index=state_index,
    vel_index=vel_index,
    freeze=args.freeze,
)
dmpn(X_train, Y_train[:, 0])
if args.cuda:
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    dmpn = dmpn.cuda()
    Y_train = Y_train.cuda()
    Y_test = Y_test.cuda()


batch_size = args.bs
learning_rate = args.lr
optimizer = torch.optim.Adam(dmpn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    t0 = time.time()
    optimizer.zero_grad()
    y_h = dmpn(X_train, state=Y_train[:, 0])
    loss = torch.mean((y_h - Y_train) ** 2)
    l = loss.detach().cpu().item()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        yh = dmpn(X_test, state=Y_test[:, 0])
        test_loss = torch.mean((yh - Y_test) ** 2)
        yht = yh.cpu().detach().numpy()
        y_h = y_h.cpu().detach().numpy()
        test_joints = Y_test.cpu().numpy()
        m = 1
        fig, axs = plt.subplots(4, 7, figsize=(30, 15))
        for k in range(4):
            for i in range(7):
                axs[k, i].plot(yht[k, :, i], label="global pol test")
                axs[k, i].plot(test_joints[k, :, i], label="test")
        plt.legend()
        plt.savefig(save_dir + "/plots.png")
        torch.save(dmpn.state_dict(), save_dir + "/policy.pt")
        print(
            "Epoch: "
            + str(epoch)
            + ", train_loss: "
            + str(l)
            + ", test_loss: "
            + str(test_loss.item())
        )
