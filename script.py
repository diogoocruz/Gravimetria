from utils import *
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.ndimage import fourier_gaussian



def main():
    print("-------")
    print("Gravimetric Data Analysis")
    print("by Diogo Cruz")
    print("-------")

    # Load data
    path = input("Enter the path to the Excel file: ")
    sheet_name = input("Enter the sheet name: ")

    try:
        data = get_data(path, sheet_name)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit()
    shp_path = input("Enter the path to the shapefile (Empty for none): ")   
    if shp_path:
        try:
            sf = get_shp(shp_path)
            print("Shapefile loaded successfully!")
        except:
            print("Error loading shapefile")
            sys.exit()
    else:
        sf = None
    # Variables
    x = input("Enter the name of the X column: ")
    y = input("Enter the name of the Y column: ")
    anomalia = input("Enter the name of the anomaly column: ")

    print("-------")
    print("Select the desired files:")
    print("1 - Scatter Plot of the Anomaly")
    print("2 - Kriging Interpolation")
    print("3 - Residual Anomaly")
    print("4 - High Pass Filter")
    print("5 - Vertical Derivative")
    print("6 - Horizontal Derivative")
    print("7 - Second Vertical Derivative")
    print("8 - Resume")
    print("9 - Exit")
    print("-------")
    inp = input("Enter the number of the desired files, separated by commas: ")

    try:
        inp_s = inp.split(",")
        inps = [int(i.strip()) for i in inp_s]
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        sys.exit()

    degree = int(input("Enter the degree of the polynomial: "))
    # Perform Kriging interpolation and residual anomaly calculation
    try:
        xi, yi, zi = krigging(data, x, y, anomalia)
        xi,yi = np.meshgrid(xi, yi)
        zi_residual, zi_regional, modelo = ajuste_polinomial(xi, yi, zi, grau=degree)

    except Exception as e:
        print(f"Error during interpolation or residual anomaly calculation: {e}")
        sys.exit()

    # Process and plot based on user selection
    for i in inps:
        try:
            if i == 1:
                plot_scatter_anomalia(data, x, y, anomalia, sf=sf)
            elif i == 2:
                plot_interpolacao(data,xi, yi, zi,x,y,anomalia, sf=sf)
            elif i == 3:
                plot_anomalia_residual(data, xi, yi, zi_residual, x, y, anomalia, sf=sf)
            elif i == 4:
                zi_hf = filtro_alta_frequencia(zi_residual, sigma=10)
                plot_alta_frquencia(xi, yi, zi_hf, sf=sf)
            elif i == 5:
                dv = derivada_vertical(zi_residual, np.diff(xi[0, :])[0], np.diff(yi[:, 0])[0])
                plot_derivada_vertical(xi, yi, dv, sf=sf)
            elif i == 6:
                dh = derivada_horizontal(xi, yi, zi_residual)
                plot_derivada_horizontal(xi, yi, dh, sf=sf)
            elif i == 7:
                sdv =segunda_derivada_vertical(zi_residual, np.diff(xi[0, :])[0], np.diff(yi[:, 0])[0])
                plot_sdv(xi, yi, sdv, sf=sf)
            elif i == 8:
                sdv =segunda_derivada_vertical(zi_residual, np.diff(xi[0, :])[0], np.diff(yi[:, 0])[0])
                dh = derivada_horizontal(xi, yi, zi_residual)
                plot_anomalias_e_derivadas(data, xi, yi, zi_residual, sdv,dh, x, y, anomalia, sf=sf)
            elif i == 9:
                print("Exiting...")
                sys.exit()
            else:
                print(f"Invalid selection: {i}")
        except Exception as e:
            print(f"Error processing selection {i}: {e}")

if __name__ == "__main__":
    main()
