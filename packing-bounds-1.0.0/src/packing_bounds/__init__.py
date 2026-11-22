"""Complete 2D/3D orthogonal knapsack upper-bound stack."""
from .api import prepare, build_model, solve, solve_all, solve_isolated
from .models import Model, STAGES
__version__ = '1.0.0'
__all__ = ['prepare','build_model','solve','solve_all','solve_isolated','Model','STAGES']
