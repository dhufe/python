import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif',style="ticks")

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

def set_size(fig):
    fig.set_size_inches(4*1.68, 4)
    fig.tight_layout()

## substrate constants
sub_heat_cond = 1.3
sub_density = 2201
sub_therm_cp = 78.75

## gas constants
gas_heat_cond = 0.026
gas_density =  1.24
gas_therm_cp = 1004
c_gas = 343

## film constants
film_density = 7120
film_therm_cp = 233

## calculating efusivities 
e_gas = np.sqrt(gas_heat_cond * gas_density * gas_therm_cp )
e_sub = np.sqrt(sub_heat_cond * sub_density * sub_therm_cp )

f = np.linspace ( 10, 1e6, 10000 )

v_th = 2 *np.pi * np.power ( 343 / f , 3.0 ) / 3

R = 10
I = 1e-5
fprf = 20
pw = 100e-9

P = I**2 / R

Qin = P * ( 1 / fprf ) / ( pw  )  

layer_thickness = np.array ( (100e-9, 200e-9, 300e-9 ))
set_style()

fig, ax = plt.subplots(1)

for iThickness in layer_thickness:
    r = e_gas / ( e_gas + e_sub + iThickness * film_density * film_therm_cp * np.sqrt( 2* np.pi* f) )
    p = r * Qin / (v_th ) 
    ax.plot ( f*1e-3, 20*np.log10(p/20e-6), label=str(int(iThickness*1e9)) + ' nm' )

ax.legend()
ax.set_xlabel( r'Frequency / kHZ' )
ax.set_ylabel( r'Sound pressure level (SPL) / dB' )
ax.set_ylim ( 0, 160 )
ax.grid(True)

set_size(fig)
plt.savefig ( 'Layerthickness_vs_Frequency_ITO.png', dpi = 300 )

plt.show()