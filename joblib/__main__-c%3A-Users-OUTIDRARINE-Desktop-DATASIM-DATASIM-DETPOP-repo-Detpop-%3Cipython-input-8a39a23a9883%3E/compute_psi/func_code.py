# first line: 1
def compute_psi(root, J, Q, downscaleFactor):
    
    n = 432
    psi = [0] * n
    
    k = 0
    progress = -1
    for root, dirnames, filenames in os.walk(root):
        for f in filenames:
            
            percentage = round(k/n * 100)
            if (percentage % 10) == 0 and percentage > progress:
                progress = percentage
                print(percentage, "%")
                
            filename = os.path.join(root, f)
            psi[k] = compute_psi_i(filename, J, Q, downscaleFactor)
            k += 1
    print("DONE")
    
    return np.array(psi)
