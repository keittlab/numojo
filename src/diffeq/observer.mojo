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

from diffeq.diffeq_traits import StepLogger

from linalg.static_matrix import (
    StaticMat as Mat,
    StaticColVec as ColVec,
    StaticRowVec as RowVec,
)


@value
struct NullLogger[m: Int](StepLogger):
    var t: List[Float64]
    var state: List[ColVec[m]]

    fn __init__(inout self):
        self.t = List[Float64]()
        self.state = List[ColVec[m]]()

    fn __copyinit__(inout self, other: Self):
        self.t = other.t
        self.state = other.state

    fn log_state[n: Int](inout self, t: Float64, s: ColVec[n]):
        constrained[m == n, "Invalid state size"]()


@value
struct StateLogger[m: Int](StepLogger):
    var t: List[Float64]
    var state: List[ColVec[m]]

    fn __init__(inout self):
        self.t = List[Float64]()
        self.state = List[ColVec[m]]()

    fn __copyinit__(inout self, other: Self):
        self.t = other.t
        self.state = other.state

    fn log_state[n: Int](inout self, t: Float64, s: ColVec[n]):
        constrained[m == n, "Invalid state size"]()
        self.state.append(rebind[ColVec[m]](s))
        self.t.append(t)
