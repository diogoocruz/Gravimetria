import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.ndimage import fourier_gaussian
from scipy.fftpack import fft2, ifft2, fftshift
from pyproj import CRS, Transformer


def get_data(path, sheet_name):
    data = pd.read_excel(path, sheet_name=sheet_name)
    return data

def get_shp(path):
    gdf = gpd.read_file(path)
        # Verificar o sistema de coordenadas original do shapefile
    original_crs = gdf.crs
    # Definir o sistema de coordenadas de destino (ETRS89)
    target_crs = CRS.from_epsg(4258)  # ETRS89

    # Criar o transformador de coordenadas
    transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)

    # Transformar as coordenadas do shapefile
    gdf = gdf.to_crs(target_crs)
    return gdf


def plot_scatter_anomalia(data, x, y, title="Anomalia", sf=None):
    plt.scatter(data[x], data[y], c=data['Anomalia'], cmap='jet', alpha=0.8)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.colorbar(label="Anomalia (uGal)")
    if sf is not None:
        sf.boundary.plot(ax=plt.gca(), edgecolor='k', linewidth=0.5)
    plt.show()

def interpolar(data, X, Y, anomalia, nx, ny, method='cubic'):
    x = data[X].values
    y = data[Y].values
    anomalia = data[anomalia].values
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x,y), anomalia, (xi, yi), method=method)
    return (xi,yi,zi)

def krigging(data, X, Y, anomalia, nx=100, ny=100, model="spherical"):
    x = data[X].values
    y = data[Y].values

    anomalia = data[anomalia].values

    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)

    OK = OrdinaryKriging(x, y, anomalia, variogram_model=model, verbose=False, enable_plotting=False)

    zi, ss = OK.execute('grid', xi, yi)
    return xi, yi, zi

def plot_interpolacao(data, xi, yi, zi, x, y, z, sf=None):
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, zi, levels=15, cmap='jet')
    plt.scatter(data[x], data[y], c=data[z], edgecolor='k', cmap='jet', alpha=0.8)
    plt.colorbar(label='Anomalia (μGal)')
    plt.title('Interpolação de Dados Gravimétricos')
    plt.xlabel('X_etrs')
    plt.ylabel('Y_etrs')
    if sf is not None:
        sf.boundary.plot(ax=plt.gca(), edgecolor='k', linewidth=0.5)
    plt.show()

def ajuste_polinomial(xi, yi, zi, grau=2):
    # Prepara os dados para o ajuste polinomial
    x_flat = xi.flatten()
    y_flat = yi.flatten()
    z_flat = zi.flatten()

    # Cria termos polinomiais
    poly_features = PolynomialFeatures(degree=grau)
    X_poly = poly_features.fit_transform(np.column_stack((x_flat, y_flat)))

    # Ajusta o modelo polinomial (anomalia regional)
    model = LinearRegression()
    model.fit(X_poly, z_flat)

    # Previsões (anomalia regional)
    z_regional = model.predict(X_poly)

    # Calcula a anomalia residual
    anomalia_residual = z_flat - z_regional

    # Reshape para o formato da grade
    zi_residual = anomalia_residual.reshape(xi.shape)
    zi_regional = z_regional.reshape(xi.shape)

    return zi_residual, zi_regional, model

def plot_anomalia_residual(data,xi, yi, zi_residual, x, y, z, sf=None):
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, zi_residual, 20, cmap='jet')
    plt.colorbar(label='Anomalia Residual (μGal)')
    plt.title('Anomalia Residual')
    plt.scatter(data[x], data[y], c=data[z], edgecolor='k', alpha=0.8, cmap='jet')
    plt.xlabel(x)
    plt.ylabel(y)
    if sf is not None:
        sf.boundary.plot(ax=plt.gca(), edgecolor='k', linewidth=0.5)
    plt.show()

def derivada_vertical(zi, dx, dy):
    # Calcula a derivada vertical usando diferenças finitas centrais
    dz_dy, dz_dx = np.gradient(zi, dy, dx)
    derivada_vertical = dz_dx + dz_dy
    return derivada_vertical

def segunda_derivada_vertical(zi, dx, dy):
    # Calcula as primeiras derivadas em relação a x e y
    dz_dx = np.gradient(zi, axis=1) / dx
    dz_dy = np.gradient(zi, axis=0) / dy

    # Calcula as segundas derivadas em relação a x e y
    d2z_dx2 = np.gradient(dz_dx, axis=1) / dx
    d2z_dy2 = np.gradient(dz_dy, axis=0) / dy
    
    # Soma das segundas derivadas para obter a segunda derivada vertical
    d2z_dz2 = d2z_dx2 + d2z_dy2
    return d2z_dz2

def derivada_vertical_fourier(zi, dx, dy):
    # Transformada de Fourier bidimensional dos dados
    zi_fft = fft2(zi)
    
    # Frequências correspondentes às dimensões x e y
    nx, ny = zi.shape
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')

    # Cálculo da derivada no domínio da frequência
    dz_fft = np.sqrt(kx**2 + ky**2) * zi_fft

    # Transformada de Fourier inversa para retornar ao domínio espacial
    dz = ifft2(dz_fft)
    
    return np.real(dz)

def plot_derivada_vertical(xi, yi, derivada_vertical,sf=None):
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, derivada_vertical, levels=15, cmap='jet')
    plt.colorbar(label='Derivada Vertical (μGal/m)')
    plt.title('Derivada Vertical da Anomalia')
    plt.xlabel('X_etrs')
    plt.ylabel('Y_etrs')
    if sf is not None:
        sf.boundary.plot(ax=plt.gca(), edgecolor='k', linewidth=0.5)
    plt.show()

def plot_sdv(xi, yi, derivada_vertical,sf=None):
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, derivada_vertical, levels=15, cmap='jet')
    plt.colorbar(label='Derivada Vertical (μGal/m^2)')
    plt.title('Segunda Derivada Vertical da Anomalia')
    plt.xlabel('X_etrs')
    plt.ylabel('Y_etrs')
    if sf is not None:
        sf.boundary.plot(ax=plt.gca(), edgecolor='k', linewidth=0.5)
    plt.show()


def filtro_alta_frequencia(zi, sigma=1):
    zi_fft = np.fft.fft2(zi)
    zi_fft_filtered = zi_fft * (1 - fourier_gaussian(np.ones_like(zi), sigma))
    zi_filtered = np.fft.ifft2(zi_fft_filtered)
    return np.real(zi_filtered)


def plot_alta_frquencia(xi,yi,zi_hf,sf=None):
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, zi_hf, levels=15, cmap='jet')
    plt.colorbar(label='Anomalia Residual de Alta Frequência (μGal)')
    plt.title('Filtro de Alta Frequência da Anomalia Residual')
    plt.xlabel('X_etrs')
    plt.ylabel('Y_etrs')
    if sf is not None:
        sf.boundary.plot(ax=plt.gca(), edgecolor='k', linewidth=0.5)
    plt.show()


def derivada_horizontal(xi,yi,zi,sf=None):
    dx = np.diff(xi[0])[0]
    dy = np.diff(yi[:,0])[0]
    dz_dy, dz_dx = np.gradient(zi, dy, dx)
    derivada_horizontal = np.sqrt(dz_dx**2 + dz_dy**2)
    return derivada_horizontal

def plot_derivada_horizontal(xi, yi, derivada_horizontal,sf=None):
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, derivada_horizontal, levels=15, cmap='jet')
    plt.colorbar(label='Derivada Horizontal (μGal/m)')
    plt.title('Derivada Horizontal da Anomalia Residual')
    plt.xlabel('X_etrs')
    plt.ylabel('Y_etrs')
    if sf is not None:
        sf.boundary.plot(ax=plt.gca(), edgecolor='k', linewidth=0.5)
    plt.show()



def plot_anomalias_e_derivadas(data,xi, yi, zi_residual, sdv, derivada_horizontal,x,y,z, cmap='jet',sf=None):
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    
    # Plot da Anomalia Residual
    ax = axes[0]
    c = ax.contourf(xi, yi, zi_residual, levels=15, cmap=cmap)
    fig.colorbar(c, ax=ax, label='Anomalia Residual (μGal)',aspect=30, pad=0.02)
    ax.set_title('Anomalia Residual')
    ax.set_xlabel('X_etrs')
    ax.set_ylabel('Y_etrs')
    ax.scatter(data[x], data[y], c=data[z], edgecolor='k', alpha=0.5, cmap=cmap, s =2)
    if sf is not None:
        sf.boundary.plot(ax=ax, edgecolor='k', linewidth=0.5)
    
    # Plot da Derivada Vertical
    ax = axes[1]
    c = ax.contourf(xi, yi, sdv, levels=15, cmap=cmap)
    fig.colorbar(c, ax=ax, label=r'Segunda Derivada Vertical (μGal/m$^2$)',aspect=30, pad=0.02)
    ax.set_title('Segunda Derivada Vertical da Anomalia')
    ax.set_xlabel('X_etrs')
    ax.set_ylabel('Y_etrs')
    ax.scatter(data[x], data[y], c=data[z], edgecolor='k', alpha=0.5,cmap=cmap, s=2)
    if sf is not None:
        sf.boundary.plot(ax=ax, edgecolor='k', linewidth=0.5)
    # Plot da Derivada Horizontal
    ax = axes[2]
    c = ax.contourf(xi, yi, derivada_horizontal, levels=15, cmap=cmap)
    fig.colorbar(c, ax=ax, label='Derivada Horizontal (μGal/m)',aspect=30, pad=0.02)
    ax.set_title('Derivada Horizontal da Anomalia')
    ax.set_xlabel('X_etrs')
    ax.set_ylabel('Y_etrs')
    ax.scatter(data[x], data[y], c=data[z], edgecolor='k', alpha=0.5,cmap=cmap, s=2)
    if sf is not None:
        sf.boundary.plot(ax=ax, edgecolor='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()