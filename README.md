# Análise de Dados Gravimétricos

Este projeto contém scripts para realizar análise de dados gravimétricos. Siga as instruções abaixo para configurar e executar o ambiente, mesmo que não tenha conhecimento prévio de Python.

## Estrutura do Diretório

- `utils.py` - Script com as funções necessárias.
- `script.py` - Script principal a ser executado.
- `requirements.toml` - Arquivo com as dependências do projeto.

## Passo a Passo para Iniciantes

### 1. Instalar Python

Primeiro, é necessário instalar o Python. Siga as instruções abaixo para o seu sistema operativo:

#### Windows

1. Aceda ao site oficial do Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Clique no botão de download para transferir o instalador do Python.
3. Execute o instalador e marque a opção "Add Python to PATH" antes de clicar em "Install Now".

#### macOS

1. Aceda ao site oficial do Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Clique no botão de download para transferir o instalador do Python.
3. Execute o instalador e siga as instruções.

#### Linux

1. Abra o terminal.
2. Execute o seguinte comando para instalar Python:
    ```sh
    sudo apt-get update
    sudo apt-get install python3 python3-pip
    ```

### 2. Instalar Pipenv

Pipenv é uma ferramenta para gerir ambientes virtuais e dependências no Python. Precisa de instalá-la para configurar o ambiente do projeto.

1. Abra o terminal (ou Prompt de Comando no Windows).
2. Execute o seguinte comando para instalar Pipenv:
    ```sh
    pip install pipenv
    ```

### 3. Configurar o Ambiente do Projeto

1. Clone o repositório para o seu diretório local:
    ```sh
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DO_REPOSITORIO>
    ```
**ou**

1. Baixe o repositório como um arquivo ZIP e extraia-o para o seu diretório local.

2. Abra o terminal e navegue até o diretório do projeto:
    ```sh
    cd <NOME_DO_REPOSITORIO>
    ```
3. Crie o ambiente virtual:
    ```sh
    python -m venv venv
    ```
4. Ative o ambiente virtual:
    (windows)
    ```sh
    venv\Scripts\activate
    ```
    (MacOs/Linux)
    ```sh
    source venv/bin/activate
    ```
5. Instale as dependências do projeto:
    ```sh
    pip install -r requirements.toml
    ```


### 4. Executar o Script

Para executar o script principal, utilize o comando abaixo e siga as instruções apresentadas no terminal:
```sh
python script.py
```


### Conseiderações

É importante que o ficheiro **Excel** tenha o seguinte formato:
* Coluna para a variável X
* Coluna para a variável Y
* Coluna com a Anomalia de Bouguer


Script desenvolvido por Diogo Filipe Gonçalves Cruz
Email: cruzgfdiogo@gmail.com
Github: https://github.com/diogoocruz
