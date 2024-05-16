<div align="center">
  <img src="docs/source/imgs/pangolinn_logo.png" width="200" alt="logo">
</div>

# pangolinn: Unit Testing for Neural Networks

As a Pangolin looks for bugs and catches them, the goal of this
library is ot help developers finding bugs in their neural networks
and newly-created models.

## 🖥 Installation

**pangolinn** is available on PyPi and can be installed by running:
```bash
pip install pangolinn
```
Alternatively, the latest development version can be installed with:
```bash
git clone https://github.com/hlt-mt/pangolinn.git
cd pangolinn
pip install -e .
```

If you want to contribute to the project, you can install
the additional development dependencies by using the `dev` specifier:

```
pip install -e .[dev]
```

## 🔧 Usage

To test your model/module you need to:

 - create a pangolinn wrapper that builds it and determines how to use it;
 - create a test suite that inherits from the pangolinn tester you want to use and
   set the attribute `module_wrapper_class` to the name of the wrapper class of your module.

For complete examples, please refer to the UTs in this repository, e.g.
[Transformer decoder causality test](tests/causal/test_causal_module_safe.py).

## 🚀 Features

The repository currently contains the following test suites:

- [x] **Encoder padding tester**: checks that the presence of padding
      does not alter the results.
- [x] **Causality tester**: checks that a module fulfils the _causal_ property,
      i.e. it does not look at future elements of the sequence (e.g., as autoregressive
      decoders have to do).


## 💡 Contributing and Feature Requests

Our goal is to provide a comprehensive test suit for neural networks, therefore contributions from interested
researchers and developers are extremely appreciated.

You can either create a ***feature request*** to propose new tests or a ***pull request*** to contribute to our
repository with your own tests.

## 📃 Licence
**pangolinn** is licensed under [Apache Version 2.0](LICENSE). 


## 🏅 Citation

If using this repository, please cite:

```
@inproceedings{papi-et-al-2024-when,
  title={{When Good and Reproducible Results are a Giant with Feet of Clay: The Importance of Software Quality in NLP}},
  author={Papi, Sara and Gaido, Marco and Pilzer, Andrea and Negri, Matteo},
  booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  address = "Bangkok, Thailand",
  year={2024}
}
```
