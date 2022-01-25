# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from typing import Union

from scipp.typing import VariableLike
import scipp as sc


def unwrap_attr(attr: VariableLike) -> Union[sc.DataArray, sc.Dataset]:
    if isinstance(attr,
                  sc.Variable) and attr.dtype in (sc.DType.DataArray, sc.DType.Dataset):
        return attr.value
    return attr
