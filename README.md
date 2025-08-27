# PROTECT: PRoxy-based individual Treatment EffeCT modeling in cancer

Official implementation of PROTECT

## installation

Development version: clone this repository, run

```bash
pip install -e .
```

To be able to run the [tutorial notebook](tutorials/inference_demo.ipynb),
also install the `dev` dependencies:

```bash
pip install -e .[dev]
```

## Usage

For applying PROTECT to your own data, see the notebook `tutorials/inference_demo.ipynb`
To run this, you'll need to install the `dev` dependencies

## data formatting

obligatory variables are

- `tx` a binary treatment indicator
- `time_cens`: a floating point variable with `time_cens = time` for patients
who had the event, and `time_cens = -time` for
patients who were censored, where:

`time_cens = (2*event - 1) * time`

you can create this using `protect.utils.time_event_to_time_cens(time,event)`
