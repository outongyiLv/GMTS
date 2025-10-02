# GMTS-Framework2

> The framework below is built on verl and reflects our modifications and the associated execution workflow.

## 🔄 Changing (1) Data reordering

We observed that, when verl enters parallel training, it does not strictly proceed group by group. Instead, it typically computes the advantage jointly and then mixes the data for gradient computation. As a first step, we implemented data reordering.

