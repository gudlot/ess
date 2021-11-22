# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Andrew R. McCluskey (arm61)
import pytest
import scipp as sc
from ess.amor import tools


def test_q_grid_linear():
    q_lin = tools.q_grid(edges=sc.array(dims=['qz'],
                                        values=[0.008, 0.08],
                                        unit='1/angstrom'),
                         scale='linear',
                         num=50)
    expected = sc.linspace(dim='qz', start=0.008, stop=0.08, num=50, unit='1/angstrom')
    assert sc.allclose(q_lin, expected)


def test_q_grid_linear_list_input():
    q_lin = tools.q_grid(edges=sc.array(dims=['qz'],
                                        values=[0.008, 0.08],
                                        unit='1/angstrom'),
                         scale=['linear'],
                         num=[50])
    expected = sc.linspace(dim='qz', start=0.008, stop=0.08, num=50, unit='1/angstrom')
    assert sc.allclose(q_lin, expected)


def test_q_grid_log():
    q_log = tools.q_grid(edges=sc.array(dims=['qz'],
                                        values=[0.008, 0.08],
                                        unit='1/angstrom'),
                         scale='log',
                         num=50)
    expected = sc.geomspace(dim='qz', start=0.008, stop=0.08, num=50, unit='1/angstrom')
    assert sc.allclose(q_log, expected)


def test_q_grid_linear_log():
    q_linlog = tools.q_grid(edges=sc.array(dims=['qz'],
                                           values=[0.008, 0.03, 0.08],
                                           unit='1/angstrom'),
                            scale=['linear', 'log'],
                            num=[16, 20])
    exp_lin = sc.linspace(dim='qz', start=0.008, stop=0.03, num=16, unit='1/angstrom')
    exp_log = sc.geomspace(dim='qz', start=0.03, stop=0.08, num=21, unit='1/angstrom')
    expected = sc.concat([exp_lin, exp_log['qz', 1:]], 'qz')
    assert sc.allclose(q_linlog, expected)


def test_q_grid_log_linear():
    q_loglin = tools.q_grid(edges=sc.array(dims=['qz'],
                                           values=[0.008, 0.03, 0.08],
                                           unit='1/angstrom'),
                            scale=['log', 'linear'],
                            num=[16, 20])
    exp_log = sc.geomspace(dim='qz', start=0.008, stop=0.03, num=16, unit='1/angstrom')
    exp_lin = sc.linspace(dim='qz', start=0.03, stop=0.08, num=21, unit='1/angstrom')
    expected = sc.concat([exp_log, exp_lin['qz', 1:]], 'qz')
    assert sc.allclose(q_loglin, expected)


def test_q_grid_linear_log_linear():
    q_linloglin = tools.q_grid(edges=sc.array(dims=['qz'],
                                              values=[0.008, 0.03, 0.08, 0.12],
                                              unit='1/angstrom'),
                               scale=['linear', 'log', 'linear'],
                               num=[16, 20, 10])
    exp_lin = sc.linspace(dim='qz', start=0.008, stop=0.03, num=16, unit='1/angstrom')
    exp_log = sc.geomspace(dim='qz', start=0.03, stop=0.08, num=21, unit='1/angstrom')
    exp_lin2 = sc.linspace(dim='qz', start=0.08, stop=0.12, num=11, unit='1/angstrom')
    expected = sc.concat([exp_lin, exp_log['qz', 1:], exp_lin2['qz', 1:]], 'qz')
    assert sc.allclose(q_linloglin, expected)


def test_q_grid_bad_input():
    with pytest.raises(ValueError):
        _ = tools.q_grid(edges=sc.array(dims=['qz'],
                                        values=[0.008, 0.03, 0.08, 0.12],
                                        unit='1/angstrom'),
                         scale=['linear', 'log'],
                         num=[16, 20])
