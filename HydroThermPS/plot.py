import matplotlib.pyplot as plt
import os
import plotly
from plotly.graph_objs import Scatter,Layout,Figure


def plot_ec_dispatch(result_dir,data,metodo):
    """Crea una gráfica del despacho económico del sistema hidrotérmico
    para el periodo de 24 h.

    Args:
        - data: generación de Potencia Activa de cada Generador Térmicos e Hidros

    Returns:
        fig: figure handle
    """
    
    if metodo == 'A':
        output_file = "Solucion Método A.html"
        titulo_plot = "DESPACHO HIDROTÉRMICO HORARIO A"
    else:
        output_file = "Solucion Método B.html"
        titulo_plot = "DESPACHO HIDROTÉRMICO HORARIO B"
        
    output_path = os.path.join(result_dir, output_file)
    
    t1 = list(range(0, 25))
    trace1 = Scatter(x=t1, y=data[0],name= "Gen Hidro 1")
    trace2 = Scatter(x=t1, y=data[1],name= "Gen Hidro 2")
    trace3 = Scatter(x=t1, y=data[2],name= "Gen Hidro 3")
    trace4 = Scatter(x=t1, y=data[3],name= "Gen Termo 1")
    trace5 = Scatter(x=t1, y=data[4],name= "Gen Termo 2")
    plotly.offline.plot({    
    "data": [trace1,trace2,trace3,trace4,trace5 ],
    "layout": Layout(title=dict(text=titulo_plot,x=0.5,
                                       font=dict(size= 35)),showlegend=True, 
                                       xaxis=dict(title=dict(text='t [h]',font=dict(size=25))),
                                       yaxis=dict(title=dict(text='P [MW]',font=dict(size=25)))
                                      
    )
    },filename=output_path )
    
#%%
def plot_dem_sup_curve(result_dir,x1,y1,x2,y2,metodo):
    
    if metodo == 'A':
        output_file = "Curva Método A.html"
        titulo_plot = "Curva de Oferta y Demanda A"
    else:
        output_file = "Curva Método B.html"
        titulo_plot = "Curva de Oferta y Demanda B"
        
    output_path = os.path.join(result_dir, output_file)
    
    trace1 = {
      "x": x1,
      "y": y1,
      "line": {"shape": 'hv'},
      "mode": 'lines',
      "name": 'value',
      "type": 'scatter'
    };
            
    trace2 = {
      "x": x2,
      "y": y2,
      "line": {"shape": 'hv'},
      "mode": 'lines',
      "name": 'value',
      "type": 'scatter'
    };
    
    data = [trace1,trace2]
    plotly.offline.plot({
        "data": data,
        "layaut": Layout(title=dict(text=titulo_plot,x=0.5,
                                       font=dict(size= 35)),showlegend=True, 
                                       xaxis=dict(title=dict(text='Precio [MWh]',font=dict(size=25))),
                                       yaxis=dict(title=dict(text='Energía [$/MWh]',font=dict(size=25)))
                                       )
    },filename=output_path)    

#%%
def plot_exp_curv(result_dir,LaM_df,metodo):
    if metodo == 'A':
        output_file = "Curvas Exportacion horario Método A.html"
        titulo_plot = "Curvas de Exportacion horario Método A"
    else:
        output_file = "Curvas Exportacion horario Método B.html"
        titulo_plot = "Curvas de Exportacion horario Método B"
        
    output_path = os.path.join(result_dir, output_file)
    
    def figures_to_html(figs, filename="Curvas Exportacion horario.html"):
        dashboard = open(filename, 'w')
        dashboard.write("<html><head></head><body><h1 style='text-align:center'>"+titulo_plot+"</h1>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
    
    
    # Example figures
    
    for i in range(len(LaM_df)):
        locals()['fig_' + str(i+1)]  = Figure(Scatter(x=LaM_df.columns, y=LaM_df.loc[i], 
              name="EXP",line_shape='hv'),Layout(title=dict(text='Barra {}'.format(i+1),x=0.5,
              font=dict(size= 35)),showlegend=False, xaxis=dict(title=dict(text='Precio [MWh]',
              font=dict(size=25))),yaxis=dict(title=dict(text='Energía [$/MWh]',font=dict(size=25)))
                                           ))
    cached = locals()
    fig = [cached[k] for k in locals() if k.startswith('fig_')]
    del cached
    figures_to_html(fig)    