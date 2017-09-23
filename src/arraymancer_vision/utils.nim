proc unsafeToTensorReshape[T](data: seq[T], shape: openarray[int]): Tensor[T] {.noSideEffect.} =
  result.shape = @shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0
  shallowCopy(result.data, data)

template unsafeAt[T](t: Tensor[T], x: int): Tensor[T] =
  t.unsafeView(x, _, _).unsafeReshape([t.shape[1], t.shape[2]])

proc unsafeSqueeze*[T](t: Tensor[T], axis: int): Tensor[T] {.noSideEffect,inline.} =
  var shape = t.shape
  assert shape[axis] == 1
  shape.delete(axis)
  t.unsafeReshape(shape)
