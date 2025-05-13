# **Smart Trajectories**

## **Funções Disponíveis**

- **`txt_to_csv(txt_filename, csv_filename)`**  
  Lê trajetórias de um arquivo de texto e grava os dados processados em um arquivo CSV.

- **`txt_to_csv_datetime(txt_filename, csv_filename)`**  
  Converte um arquivo de texto em CSV, incluindo a conversão de timestamps para objetos datetime.

- **`generate_trajectory_collection(filename)`**  
  Carrega um arquivo CSV e transforma os dados em uma coleção de trajetórias.

- **`plot_trajectories_with_background(traj_collection, background_image_path)`**  
  Plota as trajetórias sobre uma imagem de fundo estática.

- **`plot_trajectories_one_category_background(traj_collection, category, background_image_path)`**  
  Plota as trajetórias de uma categoria sobre uma imagem de fundo.

- **`plot_trajectories_with_limits(traj_collection, category, background_image_path)`**  
  Plota as trajetórias de uma categoria com a verificação de cruzamento de uma linha de referência.

- **`plot_trajectories_with_start_finish(traj_collection, category, background_image_path)`**  
  Plota as trajetórias de uma categoria verificando o início e fim correto da trajetória com linhas de partida e chegada.

- **`plot_trajectories_with_stopped(traj_collection, category, background_image_path)`**  
  Plota as trajetórias de uma categoria verificando se em algum momento ouve um comportamento de parada.

  - **`plot_trajectories_with_stop_in_rectangle(traj_collection, category, background_image_path)`**  
  Plota as trajetórias de uma categoria verificando se em algum momento ouve um comportamento de parada em uma certa área.

- **`plot_trajectories_in_monitored_area(traj_collection, category, background_image_path)`**  
  Plota as trajetórias de uma categoria e infere quais trajetórias adentraram uma certa área.
  
---

## **Como Utilizar a Biblioteca `smart-trajectories`**

1. **Instalação**  
   Instale a biblioteca usando o comando:  
   ```bash
   python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps smart-trajectories

2. Importe as funções necessárias em seu script Python. Exemplo:

```
from smart_trajectories.io import txt_to_csv
from smart_trajectories.plotting import plot_trajectories_with_background
```

3. Use as funções para converter dados, gerar coleções de trajetórias e plotar os resultados.

