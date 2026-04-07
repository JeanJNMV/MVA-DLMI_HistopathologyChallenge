# Ruche: relancer le pipeline Hibou

Ce mémo résume les commandes utiles pour relancer le pipeline sur Ruche.

## 1. Copier / mettre à jour le code depuis le Mac

Depuis le repo local:

```bash
cd "/Users/ayoub/Documents/Études supérieures/MVA/DLMI/MVA-DLMI_Histopathology_Challenge"

rsync -avP \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'data/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'logs/' \
  --exclude 'models/' \
  --exclude 'figures/' \
  --exclude 'results/' \
  ./ \
  elkbadiay@ruche.mesocentre.universite-paris-saclay.fr:/workdir/elkbadiay/MVA-DLMI_HistopathologyChallenge/
```

## 2. Se connecter à Ruche

```bash
ssh elkbadiay@ruche.mesocentre.universite-paris-saclay.fr
cd /workdir/elkbadiay/MVA-DLMI_HistopathologyChallenge
```

## 3. Préparer l'environnement

Si `.venv` n'existe pas encore:

```bash
module purge
module load python/3.14.0/gcc-15.1.0

python -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --user -U uv
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate
```

Si `.venv` existe déjà et qu'on veut juste l'utiliser:

```bash
module purge
module load python/3.14.0/gcc-15.1.0
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
```

## 4. Données et caches attendus

Le projet suppose:

```bash
/workdir/elkbadiay/MVA-DLMI_HistopathologyChallenge/data/train.h5
/workdir/elkbadiay/MVA-DLMI_HistopathologyChallenge/data/val.h5
```

Les caches utilisés par les scripts:

- `HF_HOME=/gpfs/workdir/elkbadiay/.cache/huggingface`
- `TORCH_HOME=/gpfs/workdir/elkbadiay/.cache/torch`

## 5. Smoke tests

Smoke test DINO:

```bash
sbatch scripts/smoke_test.slurm
```

Smoke test Hibou:

```bash
sbatch --export=MODEL_NAME=hibou-l scripts/smoke_test.slurm
```

Suivi:

```bash
squeue -u $USER
ls logs
cat logs/smoke_<jobid>.out
cat logs/smoke_<jobid>.err
```

## 6. LOCO light

Validation rapide du pipeline complet sur un seul fold:

```bash
sbatch --array=0 --export=MODEL_NAME=hibou-l,NUM_UNFREEZE=2,NUM_EPOCHS=2,PATIENCE=2,BATCH_SIZE=16 scripts/loco_array.slurm
```

## 7. LOCO complet

Important: `scripts/loco_cv.py` a été corrigé pour que chaque fold écrive dans un checkpoint distinct.

Avant de relancer proprement `hibou-l` avec `2` couches:

```bash
rm -f models/loco/loco_center*_hibou-l_2layers.pth
rm -f results/loco_hibou-l_2layers_center*.json
```

Puis lancer:

```bash
sbatch --export=MODEL_NAME=hibou-l,NUM_UNFREEZE=2,BATCH_SIZE=16 scripts/loco_array.slurm
```

Suivi:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName%20,State,ExitCode,Elapsed
```

Logs:

```bash
tail -n 40 logs/loco_<jobid>_0.out
tail -n 40 logs/loco_<jobid>_1.out
tail -n 40 logs/loco_<jobid>_2.out
```

Résultats:

```bash
ls results
cat results/loco_hibou-l_2layers_center0.json
cat results/loco_hibou-l_2layers_center3.json
cat results/loco_hibou-l_2layers_center4.json
```

## 8. Entraînement final

À lancer seulement après choix de la meilleure config LOCO.

Exemple:

```bash
sbatch --export=MODEL_NAME=hibou-l,NUM_UNFREEZE=2,NUM_EPOCHS=35 scripts/final_training.slurm
```

## 9. Points d'attention

- `hibou-l` attend des images redimensionnées en `224x224` avec les transforms définies dans `src/dlmi/transforms.py`.
- La branche de travail actuelle est `hibou-ruche`.
- Le dernier correctif important est le commit `5e1fc48`: isolation des checkpoints LOCO par centre tenu à l'écart.
- Si le compte Ruche est expiré, il faut contacter l'admin avant tout nouveau lancement.
