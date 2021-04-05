import os
import shutil
import HydroThermPS as HT
import pandas as pd
from pandapower import get_element_index 
from datetime import timedelta
import time

start = time.time()

# %%
input_file = 'IEEE_14_bus_system_copia.xls'  # Nombre de archivo
input_dir = 'Input_files'                 # Nombre de carpeta intertemporal
input_path = os.path.join(input_dir, input_file)
 
result_name = 'Run'
result_dir = HT.prepare_result_directory(result_name)  # Nombre + Fecha y hora

# copiar el archivo de entrada al directorio de resultados
try:
    shutil.copytree(input_path, os.path.join(result_dir, input_dir))
    
except NotADirectoryError:
    shutil.copyfile(input_path, os.path.join(result_dir, input_file))
# copiar el archivo de ejecución al directorio de resultados
shutil.copy(__file__, result_dir)

# Elegir Solver (cplex, glpk, gurobi, ...)
solver = 'glpk'

# Elegir el metodo o modelo de optimizacion 
# (A: Uninodal-OPF AC)
# (B: DC OPF - OPF AC)
metodo = 'A'

pasos_Carga_EXP = 10 #[MW]
max_Carga_EXP   = 50 #[MW]

Pij_max = [200,110,150,200,250,200,150,150,150,120,140,150,200,110,100,200,200,200,250,170]

net,load_caso_base,carga_original = HT.pandapower_model(input_path)
HT.validacion_pandapower_net (net)

G,H ,N ,A,B ,PGmax ,PGmin,BH ,PHmax ,PHmin,const,r,Vo,Vmax,Vmin,PHsubida,Qmin ,\
Qmax,PGsubida,PGbajada,PGarranque,PGparada,CA ,CP ,tablademanda,Tabla_Demanda,Upstream,\
rama,MAPG,MAPH,MAPR,Bbus,Slack,Prama_max,df_timeseries_Dem = HT.pandapower_to_pyomo_ini(input_path,metodo,Pij_max ,net)


#%% Cálculos Iterativos: Lazo Principal


# Inicio Lazo Principal [T:24 h]
LaM_df = pd.DataFrame()
Carga_EXP = 0
while Carga_EXP<=max_Carga_EXP:
    err2,FO_i,cont2 = 100, 0, 0                                         # Inicialización de las variables de control de convergencia
    load_hi = tuple(Tabla_Demanda)
    print('\n=========================== Optimizacion para las 24h ===========================\n')
    HT.printProgressBar(Carga_EXP , max_Carga_EXP, prefix = 'Progreso:', suffix = 'Completo', length = 50)
    while err2 >= 10e-6 and cont2<=10:
    #    print('\n======================================================================================\n Optimizacion para las 24h')
        FO,model,results,PHFija,GenGraph,PGini,data,in_service,LaM,RC_dict,pyomo_postprocess = HT.pyomo_model(solver,\
                    metodo,G,N,H,A,B,PGmax,PGmin,BH,PHmax,PHmin,const,r,Vo,Vmax,Vmin,PHsubida,Qmin ,\
                    Qmax,PGsubida,PGbajada,PGarranque,PGparada,CA,CP,tablademanda,Tabla_Demanda,\
                    Upstream,rama,MAPG,MAPH,MAPR,Bbus,Slack,Prama_max)
    #    print('\n======================================================================================\n')
        # Criterio de convergencia     
        err2,cont2 = HT.ErrorLazo(FO,FO_i,cont2)
        FO_i  = FO
        cont2+= 1
        print('Iteración Hidrotérmica: ',cont2)
        print('Error Opt Hidrotérmico: ', err2 , '%\n')
        
    # Inicio Lazo Interior [t : 1 h]
        res_bus,res_gen,res_line,res_trafo = [], [], [], []
        for i in range(len(Tabla_Demanda)): 
            # Lazo hora a hora
    #        print('---------------------------------------------------------\nOptimizacion para t: ',i+1,'h')
            index     = net.gen.index
            condicion = net.gen["type"] == "Ficticio"
            GenReales = index[~condicion]
            PH_PG = pd.DataFrame(list(zip(*PHFija+PGini)),columns =GenReales) 
            net.gen.p_mw[GenReales]=PH_PG.T[i][GenReales]
    
            if cont2 == 1:
                load_caso_base.p_mw.loc[0:len(load_hi[i])] = load_hi[i]
                net.load = load_caso_base.copy()
            else:
                load_caso_base.p_mw.loc[0:len(load_hi[i])] = load_hi[i]
                net.load = dd[i].copy()
            
            HT.LazoInt(net,load_caso_base,GenReales,metodo)
            if i == 0:
                red_nueva = net.load.copy()
            else: 
                red_nueva = red_nueva.append(net.load)
            res_bus.append(net.res_bus)
            res_gen.append(net.res_gen)
            res_line.append(net.res_line)
            res_trafo.append(net.res_trafo)
#            HT.printProgressBar(i + 1, len(Tabla_Demanda), prefix = 'Progress:', suffix = 'Complete', length = 50)
           
        for i in range(len(Tabla_Demanda)):
            locals()['df_' + str(i+1)] = red_nueva.iloc[len(net.bus)*i:len(net.bus)*(i+1),:]  
        Tabla_Demanda = [df_1.p_mw.values.tolist(), df_2.p_mw.values.tolist(), df_3.p_mw.values.tolist(), df_4.p_mw.values.tolist(),
                         df_5.p_mw.values.tolist(), df_6.p_mw.values.tolist(), df_7.p_mw.values.tolist(), df_8.p_mw.values.tolist(),
                         df_9.p_mw.values.tolist(), df_10.p_mw.values.tolist(),df_11.p_mw.values.tolist(),df_12.p_mw.values.tolist(),
                         df_13.p_mw.values.tolist(),df_14.p_mw.values.tolist(),df_15.p_mw.values.tolist(),df_16.p_mw.values.tolist(),
                         df_17.p_mw.values.tolist(),df_18.p_mw.values.tolist(),df_19.p_mw.values.tolist(),df_20.p_mw.values.tolist(),
                         df_21.p_mw.values.tolist(),df_22.p_mw.values.tolist(),df_23.p_mw.values.tolist(),df_24.p_mw.values.tolist()]
        
        dd = [df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,df_12,df_13,df_14,df_15,df_16,df_17,df_18,df_19,df_20,df_21,df_22,df_23,df_24]
        Tabla_Demanda_Total = [sum(tup) for tup in Tabla_Demanda]
        
        if metodo == 'A':
            tablademanda   = dict(zip(range(1,len(Tabla_Demanda)+1),Tabla_Demanda_Total))
            
        else:
            carga = []
            for i in range(len(dd)):
                aux = dd[i].p_mw.tolist()
                carga.extend(aux)                                                       # Lista de la demanda por cada hora
            load_bus  = list(net.bus.name[net.load.bus.tolist()])*(len(Tabla_Demanda)+1)# Lista de las barras en la que hay carga por cada hora
            t = [[i]*(len(net.load)) for i in range(1,len(Tabla_Demanda)+1)]            # Lista de tiempo (24h) para cada carga
            key = []
            for i in range(len(t)):
                aux = list(zip(t[i],load_bus))
                key.extend(aux)
            load_bus = dict(zip(key,carga))                                             # Diccionario de la demanda en las barras que hay cargas
        
            for k, v in tablademanda.items():
                if k in load_bus:
                    tablademanda[k]=load_bus[k]  
    
    Carga_EXP += pasos_Carga_EXP
#pyomo_postprocess()
    LaM_df[str(Carga_EXP)]=LaM
    net.load = carga_original.copy()
#    net.load.p_mw.at[pp.get_element_index(net, "load", 'Load_EXP')] = Carga_EXP
    df_timeseries_Dem[get_element_index(net, "load", 'Load_EXP')]= Carga_EXP
    tablademanda, Tabla_Demanda = HT.pandapower_to_pyomo_dem(input_file,metodo,net,df_timeseries_Dem)
#%%
runtime = time.time()-start
print ('\n\nTiempo de Ejecución:', timedelta(seconds=runtime))

list_of_dataframes = [res_bus,res_gen,res_line,res_trafo]
HT.report(result_dir,model,list_of_dataframes, metodo,runtime,cont2)

HT.plot_ec_dispatch(result_dir,data,metodo)
