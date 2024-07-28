# Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  1. Redistributions of source code must retain the above copyright
#          notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#          notice, this list of conditions and the following disclaimer in the
#          documentation and/or other materials provided with the distribution.
#
#  3. The names of its contributors may not be used to endorse or promote
#          products derived from this software without specific prior written
#          permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import dataclasses
from dataclasses import field
from typing import List, Tuple


@dataclasses.dataclass
class MersenneTwister:
    periodN: int = 624
    periodM: int = 397
    MATRIX_A: int = 0x9908B0DF
    UPPER_MASK: int = 0x80000000
    LOWER_MASK: int = 0x7FFFFFFF

    MAG: Tuple[int, int] = (0, MATRIX_A)

    mt: List[int] = field(default_factory=lambda: [0] * 624)

    mti: int = 0

    def seed(self, seed: int) -> None:
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self.periodN):
            self.mt[i] = 1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i
            self.mt[i] &= 0xFFFFFFFF  # to be under 32 bits

        self.mti = self.periodN

    def next(self) -> float:
        if self.mti == self.periodN:
            for j in range(self.periodN - self.periodM):
                y = (self.mt[j] & self.UPPER_MASK) | (self.mt[j + 1] & self.LOWER_MASK)
                self.mt[j] = self.mt[j + self.periodM] ^ (y >> 1) ^ self.MAG[y & 1]

            for j in range(self.periodN - self.periodM, self.periodN - 1):
                y = (self.mt[j] & self.UPPER_MASK) | (self.mt[j + 1] & self.LOWER_MASK)
                self.mt[j] = (
                    self.mt[j + self.periodM - self.periodN]
                    ^ (y >> 1)
                    ^ self.MAG[y & 1]
                )

            y = (self.mt[self.periodN - 1] & self.UPPER_MASK) | (
                self.mt[0] & self.LOWER_MASK
            )
            self.mt[self.periodN - 1] = (
                self.mt[self.periodM - 1] ^ (y >> 1) ^ self.MAG[y & 1]
            )

            self.mti = 0

        y = self.mt[self.mti]
        self.mti += 1

        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18

        return y / 4294967295

    def reset(self) -> None:
        self.seed(5489)
