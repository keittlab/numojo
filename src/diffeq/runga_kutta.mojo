# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Timothy H. Keitt. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from diffeq.diffeq_traits import (
    DESys,
    ExplicitRK,
    EmbeddedRK,
    StateStepper,
    StepLogger,
)

from diffeq.observer import NullLogger

from linalg.static_matrix import (
    StaticMat as Mat,
    StaticColVec as ColVec,
    StaticRowVec as RowVec,
)


struct RKFixedStepper[
    Strategy: ExplicitRK, Sys: DESys, Logger: StepLogger, n: Int
](StateStepper):
    alias StateType = ColVec[n]

    var state: Self.StateType
    var obs: Logger
    var sys: Sys

    var dt: Float64
    var t: Float64

    fn __init__(
        inout self,
        sys: Sys,
        obs: Logger,
        state: Self.StateType,
        dt: Float64,
        t0: Float64 = 0,
    ) raises:
        if len(state) != Sys.ndim():
            raise Error("Initial state has the wrong number of dimensions")
        self.sys = sys
        self.state = state
        self.obs = obs
        self.dt = dt
        self.t = t0

        self.obs.log_state(t0, state)

    fn step(inout self):
        alias m = Strategy.stages()

        var k = Mat[n, m].zeros()
        var kt = self.t + Strategy.strides[m]() * self.dt

        @parameter
        for i in range(m):
            var t = kt.get[i]()
            alias coefs = Strategy.coefs[i, m]()
            var s = self.state + k @ coefs * self.dt
            k.set_col[i](self.sys.deriv(t, s))

        alias w = Strategy.weights[m]()

        self.state += k @ w * self.dt
        self.t += self.dt

        self.obs.log_state(self.t, self.state)


struct RKAdaptiveStepper[
    Strategy: EmbeddedRK, Sys: DESys, Logger: StepLogger, n: Int
](StateStepper):
    alias StateType = ColVec[n]

    var state: Self.StateType
    var obs: Logger
    var sys: Sys

    var dt: Float64
    var t: Float64
    var tol: Float64

    fn __init__(
        inout self,
        sys: Sys,
        owned obs: Logger,
        state: Self.StateType,
        dt: Float64,
        t0: Float64 = 0,
        tol: Float64 = 1e-9,
    ) raises:
        if len(state) != Sys.ndim():
            raise Error("Initial state has the wrong number of dimensions")
        self.sys = sys
        self.state = state
        self.obs = obs^
        self.tol = tol
        self.dt = dt
        self.t = t0

        self.obs.log_state(t0, state)

    fn step(inout self):
        alias p = Strategy.order2()
        alias w1 = Strategy.weights[m]()
        alias w2 = Strategy.weights2[m]()
        alias m = Strategy.stages()

        var k = Mat[n, m].zeros()
        var kt = self.t + Strategy.strides[m]() * self.dt

        @parameter
        for i in range(m):
            var t = kt.get[i]()
            alias coefs = Strategy.coefs[i, m]()
            var s = self.state + k @ coefs * self.dt
            k.set_col[i](self.sys.deriv(t, s))

        alias dw = w1 - w2
        var err = k @ dw * self.dt

        if err.max_value() < self.tol:
            self.state += k @ w1 * self.dt
            self.t += self.dt

            self.obs.log_state(self.t, self.state)

        var s = (self.tol / err.max_value() / 2) ** (1 / p)
        self.dt *= max(min(s, 4), 1 / 4)


""" from diffeq.desys_examples import Lorenz
from diffeq.rk_strategies import RK4, RK45, LStable
from diffeq.observer import NullLogger, StateLogger


fn main() raises:
    var grad = Lorenz(10, 28, 8 / 3)
    var s0 = ColVec[3](2.0, 1.0, 1.0)
    var obs = StateLogger[3]()
    var stepper = RKAdaptiveStepper[RK45](grad, obs^, s0, 0.01)
    for _ in range(30):
        stepper.step()

    for i in range(len(stepper.obs.t)):
        print("t =", stepper.obs.t[i], end=": ")
        for j in range(3):
            print(stepper.obs.state[i][j], end=" ")
        print() """
