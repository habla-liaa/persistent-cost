import time
import numpy as np
from persistent_cost.cylinder import cylinder_pipeline
# import cProfile
# import pstats
# import io

def main():
    # Random 50 points cloud X, Y, f range(25)
    np.random.seed(42)
    Y = np.random.rand(50, 2) * 10
    X = Y[:25]
    f = np.arange(25)

    maxdim = 1
    threshold = 5

    # only profile this part
    # pr = cProfile.Profile()
    # pr.enable()    
    # tic
    tic = time.time()
    d_ker, d_coker = cylinder_pipeline(X, Y, f, threshold, maxdim, verbose=True)
    toc = time.time()
    print(f"Cylinder persistence computed in {toc - tic:.4f} seconds.")
    # toc
    # pr.disable()
    # s = io.StringIO()
    # sortby = pstats.SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # # save profile results to file
    # with open("profile_results.txt", "w") as f:
    #     f.write(s.getvalue())

if __name__ == "__main__":
    main()