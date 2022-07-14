### **Microbes API built using FastAPI**<br>

- Uses `python-3.8.13`
- API built using FastAPI
- Locally hosted at `http://127.0.0.1:4004`
- Functionalities Present:
    - `/version`
    - `/distribution/<feature_name>`
    - `train`
    - `train/logs`
    - `train/logs/<model_name>/<fold>`
    - `infer`

### **Detailed Information**

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4. Run `start-api-server.bat` (or setup `.vscode`).
5. The API will now be served at `http://127.0.0.1:4004`

**OR**

1. ~~Pull the docker image using `docker pull prashanthacsq/microbes-api`~~
2. Run `docker-run.bat`. 
3. The API will now be served at `http://127.0.0.1:4004`

<br>