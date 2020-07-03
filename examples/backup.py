    def forward(self, inputs):
        psi_s, cost = self.psi(torch.cat([self.data, inputs], dim=0))
        self.odefunc.psi_s = psi_s
        mb_cat = psi_s.shape[0]
        z_shape = (mb_cat, self.num_samples)
        z = torch.zeros(z_shape, device = psi_s.device, dtype = psi_s.dtype)
        interval = self.interval
        zs = odeint(self.odefunc, z, interval)
        c_s_list = [torch.matmul(self.odefunc.c_s, self.odefunc.get_weights(time))
                    for time in interval]
        psi_s_list = [torch.matmul(psi_s[0:self.num_samples, :], self.odefunc.get_weights(time))
                      for time in interval]
        vals_list = [torch.matmul(z[0:self.num_samples, :], c)
                     for z, c in zip(zs, c_s_list)]
        u_list = [val + psi for val, psi in zip(vals_list, psi_s_list)]

        for u, c in zip(u_list, c_s_list):
            cost += flat_dot(u, c) / self.max_freq

        # res = torch.cat(u_list, dim=1)[self.num_samples:, :]
        print(cost)
        return torch.cat(list(zs), dim=1)[self.num_samples:, :], cost
