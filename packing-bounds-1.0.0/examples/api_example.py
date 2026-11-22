from pathlib import Path
import json
from packing_bounds import solve, build_model

if __name__ == '__main__':
    instance=json.loads(Path(__file__).with_name('seven_squares_2d.json').read_text())
    print(solve(instance,stage='full',mode='ip',time_limit=30))
    model=build_model(instance,'full')
    print('Sparse model:',model.A.shape,'with',model.A.nnz,'nonzero coefficients')
