"""Backend-neutral ``keras.ops`` + HGQ2 building blocks of the CLIC MaskFormer.

Rules: no module in this package imports torch, and this ``__init__`` imports no submodule. The backend is chosen
by ``KERAS_BACKEND`` before keras is first imported; this package does not set it, because its modules run under
the torch-backend hybrid (``hepattn.keras``, which pins torch) and are meant to run unchanged under other backends.
"""
