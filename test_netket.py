import netket as nk

g = nk.graph.Chain(length=8)

hilbert = nk.hilbert.Spin(
    s=1/2,
    N=g.n_nodes,
)

H = nk.operator.Ising(
    hilbert,
    graph=g,
    h=1.0,
)

model = nk.models.RBM()

sampler = nk.sampler.MetropolisLocal(
    hilbert,
)

vs = nk.vqs.MCState(
    sampler,
    model,
    n_samples=1024,
)

gs = nk.VMC(
    H,
    nk.optimizer.Sgd(learning_rate=0.05),
    variational_state=vs,
)

gs.run(20)

