template unsafeAt[T](t: Tensor[T], x: int): Tensor[T] =
  t[x, _, _].reshape([t.shape[1], t.shape[2]])
