template unsafeAt[T](t: Tensor[T], x: int): Tensor[T] =
  t.unsafeSlice(x, _, _).unsafeReshape([t.shape[1], t.shape[2]])
