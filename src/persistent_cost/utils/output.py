def print_diagram(diagram):
    for dim, dgm in enumerate(diagram):
        print(f"Dimension {dim}:")
        if len(dgm) == 0:
            print("  No points")
        else:
            for point in dgm:
                print(f"  {point}")

