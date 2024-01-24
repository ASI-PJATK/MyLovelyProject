# My Lovely Project

## Setting Up the Project Environment
### Prerequisites

- **Conda**: Ensure that you have Conda installed on your system. If you do not have Conda installed, you can download and install Miniconda (a minimal installer for Conda) from [here](https://docs.conda.io/en/latest/miniconda.html).

## Setting up the enviroment
- create the conda environment with required dependencies `conda env create -f environment.yml`
- activate virtual env `conda activate my_project_env` 
- custom package can be installed via `pip install lovely-pancake`
- hosing.csv file stored in Azure Blob Storage. Use `dvc pull` to get it (sas token will get disabled on 01/04/2024)
- execute the pipeline `kedro run` 

## Building and running docker container
- navigate to main folder of the project
- use command `docker build -f Dockerfile -t test`
- if you already build the container use `docker run` instead

## Runing the docker compose
In main directory of the project
-  use command `docker compose up --build` when running docer for the first time
- use command `docker compose up` with docker image already built
Than to test if everything is working check the fastAPI at `127.0.0.1:8000/docs`

## RestAPI Endpoint
- navigate to restAPI folder `cd src/fastAPI/`
- run the restAPI `uvicorn main:app`
- example endpoint calls :
```html
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "longitude": -119.85,
  "latitude": 36.77,
  "housing_median_age": 10,
  "total_rooms": 1000,
  "total_bedrooms": 200,
  "population": 500,
  "households": 180,
  "median_income": 3.5000
}'

```
```html
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "longitude": -117.03,
  "latitude": 32.71,
  "housing_median_age": 22,
  "total_rooms": 4500,
  "total_bedrooms": 700,
  "population": 2300,
  "households": 600,
  "median_income": 6.3000
}'
```


## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`


## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will `pip-compile` the contents of `src/requirements.txt` into a new file `src/requirements.lock`. You can see the output of the resolution by opening `src/requirements.lock`.

After this, if you'd like to update your project requirements, please update `src/requirements.txt` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r src/requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
