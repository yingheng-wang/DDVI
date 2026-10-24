# Third-Party Notices

This repository is released under the MIT License ([LICENSE](LICENSE)), except for the file listed under GPL-3.0
below. The following components are adapted from other projects.

| Component | Source | License |
|---|---|---|
| `ddvi/flows.py`: masked autoregressive network (MADE) | [karpathy/pytorch-made](https://github.com/karpathy/pytorch-made) | MIT |
| `ddvi/flows.py`: inverse autoregressive flow layer | [pclucas14/iaf-vae](https://github.com/pclucas14/iaf-vae) | MIT |
| `ddvi/priors.py`: Gaussian kernel density estimate | [EugenHotaj/pytorch-generative](https://github.com/EugenHotaj/pytorch-generative) | MIT |
| `ddvi/models/aae.py`: adversarial autoencoder baselines | [fducau/AAE_pytorch](https://github.com/fducau/AAE_pytorch) | GPL-3.0 |

## GPL-3.0

`ddvi/models/aae.py` follows the adversarial autoencoder implementation of AAE_pytorch and is distributed under the
GNU General Public License v3.0; the full text is in [licenses/GPL-3.0.txt](licenses/GPL-3.0.txt).

## MIT

pytorch-made: Copyright (c) Andrej Karpathy.
iaf-vae: Copyright (c) 2020 Lucas Caccia.
pytorch-generative: Copyright (c) 2020 Eugen Hotaj.

These components are provided under the following license:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
