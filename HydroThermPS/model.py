from __future__ import division
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.opt.base import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import pandas as pd
import pandapower as pp
import sys
import os
from itertools import compress,product


#%% 
def pandapower_model(input_file):
    """Lee el archivo de entrada de Excel Read Excel 
    Lee la correspondiente hoja de Excel Reads the Excel para 
    la MODELACIÓN DE LA RED EN PANDAPOWER (Flujo Optimo de Potencia AC) .

    Args:
        - input_file: nombre del archivo con hojas de cálculo de Excel
        
    Returns:
        - net            : red modelada en pandapower (Barras, Cargas, 
                           Generadores, Lineas, Transformadores, Compensacion, Costos)
        - load_caso_base : Copia del modelo de las cargas inicial para calculos multiperiodo
        - carga_original : Copia del modelo de las cargas inicial para calculos multiperiodo
    """ 
    input_df = pd.ExcelFile(input_file)
    
    input_sheet_names = input_df.sheet_names
    
    net = pp.create_empty_network(f_hz=60.0)
    
    for sheet_name in input_sheet_names:
        if sheet_name == 'BUS':
            df = input_df.parse(sheet_name=sheet_name,index_col=0)
            for idx in df.index:
                pp.create_bus(net,vn_kv=df.at[idx,"vn_kv"],name=df.at[idx,"name"],type=df.at[idx,"type"],
                              zone=df.at[idx,"zone"],in_service=df.at[idx,"in_service"],
                              max_vm_pu=df.at[idx,"max_vm_pu"],min_vm_pu=df.at[idx,"min_vm_pu"])
        elif sheet_name == 'LOAD':
            df = input_df.parse(sheet_name=sheet_name,index_col=0) 
            for idx in df.index:
                bus_idx = pp.get_element_index(net, "bus", df.bus[idx])
                pp.create_load(net,bus=bus_idx,p_mw=df.at[idx,"p_mw"],q_mvar=df.at[idx,"q_mvar"],
                               const_z_percent=df.at[idx,"const_z_percent"],const_i_percent=df.at[idx,"const_i_percent"],
                               sn_mva=df.at[idx,"sn_mva"],name=df.at[idx,"name"],scaling=df.at[idx,"scaling"],
                               in_service=df.at[idx,"in_service"],type=df.at[idx,"type"],controllable=df.at[idx,"controllable"])
            load_caso_base = net.load.copy()
            carga_original = net.load.copy()
        elif sheet_name == 'GEN':
            df = input_df.parse(sheet_name=sheet_name,index_col=0)
            for idx in df.index:
                gen_idx = pp.get_element_index(net, "bus", df.bus[idx])
                pp.create_gen(net,bus=gen_idx,p_mw=df.at[idx,"p_mw"],vm_pu=df.at[idx,"vm_pu"],
                              sn_mva=df.at[idx,"sn_mva"],name=df.at[idx,"name"],
                              max_q_mvar=df.at[idx,"max_q_mvar"],min_q_mvar=df.at[idx,"min_q_mvar"],
                              min_p_mw=df.at[idx,"min_p_mw"],max_p_mw=df.at[idx,"max_p_mw"],
                              scaling=df.at[idx,"scaling"],type=df.at[idx,"type"],slack=df.at[idx,"slack"],
                              controllable=df.at[idx,"controllable"],in_service=df.at[idx,"in_service"])         
        elif sheet_name == 'LINE':
            df = input_df.parse(sheet_name=sheet_name,index_col=0)
            for idx in df.index:
                From_bus = pp.get_element_index(net, "bus", df.from_bus[idx])
                To_bus   = pp.get_element_index(net, "bus", df.to_bus[idx])
                pp.create_line_from_parameters(net,from_bus=From_bus,to_bus=To_bus,length_km=df.at[idx,"length_km"],
                                               r_ohm_per_km=df.at[idx,"r_ohm_per_km"],x_ohm_per_km=df.at[idx,"x_ohm_per_km"],
                                               c_nf_per_km=df.at[idx,"c_nf_per_km"],max_i_ka=df.at[idx,"max_i_ka"],
                                               type=df.at[idx,"type"],in_service=df.at[idx,"in_service"],
                                               df=df.at[idx,"df"],parallel=df.at[idx,"parallel"],g_us_per_km=df.at[idx,"g_us_per_km"],
                                               name=df.at[idx,"name"])        
        elif sheet_name == 'TRAFO':
            df = input_df.parse(sheet_name=sheet_name,index_col=0)
            for idx in df.index:
                hv_bus = pp.get_element_index(net, "bus", df.hv_bus[idx])
                lv_bus = pp.get_element_index(net, "bus", df.lv_bus[idx])
                pp.create_transformer_from_parameters(net,hv_bus=hv_bus,lv_bus=lv_bus,sn_mva=df.at[idx,"sn_mva"],
                                                      vn_hv_kv=df.at[idx,"vn_hv_kv"],vn_lv_kv=df.at[idx,"vn_lv_kv"],
                                                      vkr_percent=df.at[idx,"vkr_percent"],vk_percent=df.at[idx,"vk_percent"],
                                                      pfe_kw=df.at[idx,"pfe_kw"],i0_percent=df.at[idx,"i0_percent"],
                                                      shift_degree=df.at[idx,"shift_degree"],tap_side=df.at[idx,"tap_side"],
                                                      tap_neutral=df.at[idx,"tap_neutral"],tap_max=df.at[idx,"tap_max"],
                                                      tap_min=df.at[idx,"tap_min"],tap_step_percent=df.at[idx,"tap_step_percent"],
                                                      tap_step_degree=df.at[idx,"tap_step_degree"],tap_pos=df.at[idx,"tap_pos"],
                                                      tap_phase_shifter=df.at[idx,"tap_phase_shifter"],in_service=df.at[idx,"in_service"],
                                                      parallel=df.at[idx,"parallel"],df=df.at[idx,"df"],name=df.at[idx,"name"])
                
        elif sheet_name == 'SHUNT':
            df = input_df.parse(sheet_name=sheet_name,index_col=0) 
            for idx in df.index:
                bus_idx = pp.get_element_index(net, "bus", df.bus[idx])
                pp.create_shunt(net,bus=bus_idx,p_mw=df.at[idx,"p_mw"],q_mvar=df.at[idx,"q_mvar"],
                               vn_kv=df.at[idx,"vn_kv"],name=df.at[idx,"name"],step=df.at[idx,"step"],
                               in_service=df.at[idx,"in_service"],max_step=df.at[idx,"max_step"]) 
        
        elif sheet_name == 'COST':
            df = input_df.parse(sheet_name=sheet_name,index_col=0)
            for idx in df.index:
                try:
                    gen_idx = pp.get_element_index(net, "gen", df.element[idx])
                except:
                    try:
                        gen_idx = pp.get_element_index(net, "sgen", df.element[idx])
                    except:
                        try:
                            gen_idx = pp.get_element_index(net, "ext_grid", df.element[idx])
                        except:
                            raise TypeError('No existe el elemento')
                pp.create_poly_cost(net,element=gen_idx,et=df.at[idx,"et"],cp1_eur_per_mw=df.at[idx,"cp1_eur_per_mw"],cp0_eur=df.at[idx,"cp0_eur"],
                                    cq1_eur_per_mvar=df.at[idx,"cq1_eur_per_mvar"],cq0_eur=df.at[idx,"cq0_eur"],
                                    cp2_eur_per_mw2=df.at[idx,"cp2_eur_per_mw2"],cq2_eur_per_mvar2=df.at[idx,"cq2_eur_per_mvar2"])
            
        elif sheet_name == 'BUS_GEODATA':
            df = input_df.parse(sheet_name=sheet_name,index_col=0)
            net.bus_geodata[:]=df
    return net,load_caso_base,carga_original 

#%%
def validacion_pandapower_net (net) :   
    dct  = pp.diagnostic(net, report_style='detailed', warnings_only=True)
    if bool(not dct):
        print('Ningun error en la creación de la red')
    else:
        print('¡Error en la creación de la red!')
        print(dct)
        sys.exit(1)


#%% 
def pandapower_to_pyomo_ini(input_file,metodo,Pij_max,net):
    '''Se realiza las adaptacion de los datos del modelo de PANDAPOWER
    y se leen datos adicionales del archivo de entrada de Excel, para crear el 
    formato de entrada del modelo de PYOMO.

    Args:
        - input_file: nombre del archivo con hojas de cálculo de Excel
        - net            : red modelada en pandapower (Barras, Cargas, 
                           Generadores, Lineas, Transformadores, Compensacion, Costos)
    Returns:
        - G           : Nombres Generadores termoeléctricos
        - H           : Nombres Generadores hidroeléctricos
        - N           : Nombres de las Barras del sistema  
        - A           : Parámetro A de la función de Costo [$] 
        - B           : Parámetro B de la función de Costo [$/MW] Generadores Térmicos
        - PGmax       : Limite máximo de potencia térmica generada [MW]
        - PGmin       : Limite mínimo de potencia térmica generada [MW]
        - BH          : Parámetro B de la función de Costo [$/MW] Generadores Hidros
        - PHmax       : Limite máxima de potencia hidro generada [MW]
        - PHmin       : Limite mínima de potencia hidro generada [MW]
        - MAPH        : Generadores Hidroeléctricos - Barras
        - MAPG        : Generadores Termoeléctricos - Barras
        - MAPR        : Ramas - Barra i - Barra j 
        - const       : Constante de eficiencia [MWs/m^3]
        - r           : Influjos naturales [m^3/s]
        - Vo          : Volumen inicial del enbalse [Hm^3]
        - Vmax        : Volumen maximo del embalse [Hm^3]
        - Vmin        : Volumen minimo del embalse [Hm^3]
        - PHsubida    : Rampa de subida para generadores hidros [MW]
        - Qmin        : Caudal minimo turbinado [m^3/s]
        - Qmax        : Caudal maximo turbinado [m^3/s]
        - PGsubida    : Rampa de subida para generadores termos [MW]
        - PGbajada    : Rampa de bajada para generadores termos [MW]
        - PGarranque  : Rampa de arranque para generadores termos[MW]
        - PGparada    : Rampa de parada para generadores termos [MW]
        - CA          : Costo fijo de arranque de generadores termo [$/MWh]
        - CP          : Costo fijo de parada de generadores termo [$/MWh]
        - tablademanda: Tabla de demanda horaria [MW]
        - Upstream    : Topología de la red hidrológica
        - Bbus        : Tabla de Susceptancia
        - Slack       : Nombre de la barra Slack
        - rama        : Nombres de las ramas del sistema
        - Prama_max   : Limite máximo de potencia por la rama ij
    '''

    index       = net.gen.index
    condition1   = net.gen["type"] == "Hidro"
    condition2   = net.gen["type"] == "Termico"
    GenTermo    = index[condition2]
    GenHidro    = index[condition1]
    GT_Idx_list = GenTermo.tolist()
    GH_Idx_list = GenHidro.tolist()
    G = net.gen.name.loc[GT_Idx_list].tolist()                         
    H = net.gen.name.loc[GH_Idx_list].tolist()                         
                                                                         
    values = net.poly_cost.cp0_eur.loc[GT_Idx_list].tolist()
    A      = dict(zip(G,values))                                       
    values = net.poly_cost.cp1_eur_per_mw.loc[GT_Idx_list].tolist()
    B      = dict(zip(G,values))                                       
    values = net.gen.max_p_mw.loc[GT_Idx_list].tolist()
    PGmax  = dict(zip(G,values))                                       
    values = net.gen.min_p_mw.loc[GT_Idx_list].tolist()
    PGmin  = dict(zip(G,values))                                       
    
    values = net.poly_cost.cp1_eur_per_mw.loc[GH_Idx_list].tolist()
    BH     = dict(zip(H,values))
    values = net.gen.max_p_mw.loc[GH_Idx_list].tolist()
    PHmax  = dict(zip(H,values))                                       
    values = net.gen.min_p_mw.loc[GH_Idx_list].tolist()
    PHmin  = dict(zip(H,values))                                           
    
    df_test = pd.read_excel(input_file,sheet_name="GHidro",index_col=0)
    df_test.columns = ['ref', 'const','r','Vo','Vmax','Vmin','PHsubida','Qmin','Qmax']
    keys   = df_test.ref.tolist()
    
    values = df_test.const.tolist()
    const  = dict(zip(keys,values))                                    
    values = df_test.r.tolist()
    
    df = pd.read_excel(input_file,sheet_name="Influjos",index_col=0) 
    rr = df.T.values.tolist()
    r  = dict(zip(H,rr))                                              
    values = df_test.Vo.tolist()
    Vo  = dict(zip(keys,values))                                       
    values = df_test.Vmax.tolist()
    Vmax  = dict(zip(keys,values))                                     
    values = df_test.Vmin.tolist()
    Vmin  = dict(zip(keys,values))                                    
    values = df_test.PHsubida.tolist()
    PHsubida  = dict(zip(keys,values))                                 
    values = df_test.Qmin.tolist()
    Qmin  = dict(zip(keys,values))                                    
    values = df_test.Qmax.tolist()
    Qmax  = dict(zip(keys,values))                                   
    
    df_test = pd.read_excel(input_file,sheet_name="GTerm",index_col=0)
    df_test.columns = ['ref', 'PGsubida','PGbajada','PGarranque','PGparada','CA','CP']
    
    values = df_test.PGsubida.tolist()
    PGsubida  = dict(zip(G,values))                                    
    values = df_test.PGbajada.tolist()
    PGbajada  = dict(zip(G,values))                                    
    values = df_test.PGarranque.tolist()
    PGarranque  = dict(zip(G,values))                                  
    values = df_test.PGparada.tolist()
    PGparada  = dict(zip(G,values))                                    
    values = df_test.CA.tolist()
    CA  = dict(zip(G,values))                                         
    values = df_test.CP.tolist()
    CP  = dict(zip(G,values))                                          
    
    df_timeseries_Dem   = pd.read_excel(input_file,sheet_name="Demanda",index_col=0) 
    Tabla_Demanda       = df_timeseries_Dem.values.tolist()
    Tabla_Demanda_Total = [sum(tup) for tup in Tabla_Demanda]
    if metodo == 'A':
        tablademanda        = dict(zip(range(1,len(Tabla_Demanda)+1),Tabla_Demanda_Total))
    elif metodo == 'B':
        MAPG ,MAPH = [], []
        N    = net.bus.name.tolist()  
        MAPH = net.bus.name.loc[net.gen.bus[GH_Idx_list]]
        MAPH = list(zip(H,MAPH))                          
        MAPG = net.bus.name.loc[net.gen.bus[GT_Idx_list]]
        MAPG = list(zip(G,MAPG))                          	
        #  Creacion de la tabla de Demanda
        t = [[i]*(len(net.bus)) for i in range(1,len(Tabla_Demanda)+1)]
        key = []
        for i in range(len(t)):
            aux = list(zip(t[i],net.bus.name.tolist()))
            key.extend(aux)
        tablademanda = dict.fromkeys(key,0)
        
        Dem_split = [j for i in Tabla_Demanda for j in i]
        load_bus  = list(net.bus.name[net.load.bus.tolist()])*(len(Tabla_Demanda)+1)
        t = [[i]*(len(net.load)) for i in range(1,len(Tabla_Demanda)+1)]
        key = []
        for i in range(len(t)):
            aux = list(zip(t[i],load_bus))
            key.extend(aux)
        load_bus = dict(zip(key,Dem_split))
        
        for k, v in tablademanda.items():
            if k in load_bus:
                tablademanda[k]=load_bus[k]
                
        # Creación de la Tabla de Susceptancia
        To_bus, From_bus, S_branch = [],[],[]
        for i in range(0,len(net.line)+len(net.trafo)):
            From_bus.append(net.bus.name[(net._ppc['branch'][i][0]).real])
            To_bus.append(net.bus.name[(net._ppc['branch'][i][1]).real])
            if i <= (len(net.line)-1):
                S_branch.append(1/((net._ppc['branch'][i][3].real)*100))
            else:
                S_branch.append(1/((net._ppc['branch'][i][3].real)*net.trafo.sn_mva[i-len(net.line)]))
        S_branch[6]  = S_branch[6]*10                                          # Correccion de valor ramam 3-4 segun paper 
        keys1 = tuple(zip(From_bus,To_bus))
        keys2 = tuple(zip(To_bus,From_bus))
        Branch_tupla1 = tuple(zip(keys1,S_branch))
        Branch_tupla2 = tuple(zip(keys2,S_branch))
        Branch_data1 = {x:0 for x, _ in Branch_tupla1} 
        for name, num in Branch_tupla1: Branch_data1[name] += num 
        Branch_data2 = {x:0 for x, _ in Branch_tupla2} 
        for name, num in Branch_tupla2: Branch_data2[name] += num 
        Branches_data = {**Branch_data1, **Branch_data2}
        
        Bbus_key = list(product(net.bus.name.tolist(),net.bus.name.tolist()))
        Bbus = dict.fromkeys(Bbus_key,0)
        for k, v in Bbus.items():
            if k in Branches_data:
                Bbus[k]=Branches_data[k]
        
        for i in range(len(net.gen.slack)):
            if net.gen.slack[i]==True :
                Slack = net.bus.name[net.gen.bus[i]]                                        # Nombre de la barra Slack     
        
        rama = net.line.name.tolist()+net.trafo.name.tolist()
        rama = list(dict.fromkeys([rama[i].partition("/")[0] for i in range(len(rama))]))
                                                                               # Nombre de las ramas del sistema
        From_bus, To_bus = [], []
        for i in range(0,len(net.line)):
            From_bus.append(net.bus.name[net.line.from_bus[i]])
            To_bus.append(net.bus.name[net.line.to_bus[i]])
        To_bus.pop(0)                                                          # Barra i de la rama
        From_bus.pop(0)                                                        # Barra j de la rama
        for i in range(0,len(net.trafo)):
            From_bus.append(net.bus.name[net.trafo.hv_bus[i]])
            To_bus.append(net.bus.name[net.trafo.lv_bus[i]])    
        MAPR = list(zip(rama,From_bus,To_bus))                                 # Rama - Barra i - Barra j
        
        Prama_max = dict(zip(rama,Pij_max))                                    # Limite maximo de potencia por la rama ij 
    else:
        raise ValueError("El metodo ingresado {} no corresponde a las opciones disponibles",format(metodo))
    
    df       = pd.read_excel(input_file,sheet_name="UPSTREAM",index_col=0) 
    Upstream = df.values.tolist()
    for i in range(len(Upstream)):    
        Upstream[i] = list(compress(df.columns,Upstream[i]))
    Upstream = dict(zip(H,Upstream))    

    def prueba():
        print('Si se ha podido')

    if metodo == 'A':
        N,rama,MAPG,MAPH,MAPR,Bbus,Slack,Prama_max = [],[],[],[],[],{},'',{}
        return G,H ,N ,A,B ,PGmax ,PGmin,BH ,PHmax ,PHmin,const,r,Vo,Vmax,Vmin,PHsubida,Qmin ,\
               Qmax,PGsubida,PGbajada,PGarranque,PGparada,CA ,CP ,tablademanda,Tabla_Demanda,Upstream,\
               rama,MAPG,MAPH,MAPR,Bbus,Slack,Prama_max,df_timeseries_Dem
    else:
        return G,H ,N ,A,B ,PGmax ,PGmin,BH ,PHmax ,PHmin,const,r,Vo,Vmax,Vmin,PHsubida,Qmin ,\
               Qmax,PGsubida,PGbajada,PGarranque,PGparada,CA ,CP ,tablademanda,Tabla_Demanda,Upstream,\
               rama,MAPG,MAPH,MAPR,Bbus,Slack,Prama_max,df_timeseries_Dem
#%%
def pandapower_to_pyomo_dem(input_file,metodo,net,df_timeseries_Dem):
    '''Se realiza las adaptacion de los datos del modelo de PANDAPOWER
    y se leen datos adicionales del archivo de entrada de Excel, para crear el 
    formato de entrada del modelo de PYOMO.

    Args:
        - input_file: nombre del archivo con hojas de cálculo de Excel
        - metodo    : Método de solución A o B
        - net       : red modelada en pandapower (Barras, Cargas, 
                           Generadores, Lineas, Transformadores, Compensacion, Costos)
    Returns:
        - tablademanda: Tabla de demanda horaria [MW]
 
'''
    Tabla_Demanda       = df_timeseries_Dem.values.tolist()
    Tabla_Demanda_Total = [sum(tup) for tup in Tabla_Demanda]
    if metodo == 'A':
        tablademanda        = dict(zip(range(1,len(Tabla_Demanda)+1),Tabla_Demanda_Total))
    elif metodo == 'B':
        #  Creacion de la tabla de Demanda
        t = [[i]*(len(net.bus)) for i in range(1,len(Tabla_Demanda)+1)]
        key = []
        for i in range(len(t)):
            aux = list(zip(t[i],net.bus.name.tolist()))
            key.extend(aux)
        tablademanda = dict.fromkeys(key,0)
        
        Dem_split = [j for i in Tabla_Demanda for j in i]
        load_bus  = list(net.bus.name[net.load.bus.tolist()])*(len(Tabla_Demanda)+1)
        t = [[i]*(len(net.load)) for i in range(1,len(Tabla_Demanda)+1)]
        key = []
        for i in range(len(t)):
            aux = list(zip(t[i],load_bus))
            key.extend(aux)
        load_bus = dict(zip(key,Dem_split))
        
        for k, v in tablademanda.items():
            if k in load_bus:
                tablademanda[k]=load_bus[k]
    else:
        raise ValueError("El metodo ingresado {} no corresponde a las opciones disponibles",format(metodo))
    return tablademanda, Tabla_Demanda

#%%             

def pyomo_model(solver,metodo,G,N,H,A,B,PGmax,PGmin,BH,PHmax,PHmin,const,r,Vo,Vmax,Vmin,PHsubida,Qmin ,\
                Qmax,PGsubida,PGbajada,PGarranque,PGparada,CA,CP,tablademanda,Tabla_Demanda,Upstream,\
                rama,MAPG,MAPH,MAPR,Bbus,Slack,Prama_max):
    model   = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT)
           
    # CONJUNTOS
    model.G  = Set(initialize = G, doc = 'Generadores termoeléctricos') 
    model.H  = Set(initialize = H, doc = 'Generadores hidroeléctricos')
    model.T  = RangeSet(0,24)
    
    if metodo == 'B':
        model.L    = Set(initialize = rama, doc = 'Lineas de transmision y Transformadores')
        model.N    = Set(initialize =N, doc ='Nodos')
        model.NP   = SetOf(model.N)
        model.MAPG = Set(within = model.G*model.N, ordered = True, initialize=MAPG, doc = 'Generadores termoeléctricos - Barras')
        model.MAPH = Set(within = model.H*model.N, ordered = True, initialize=MAPH, doc = 'Generadores hidroeléctricos -Barras')
        model.MAPL = Set(within = model.L*model.N*model.N, ordered = True, initialize=MAPR, doc = 'Ramas -Barra i - Barra j')

    # PARAMETROS
    model.A          = Param(model.G, initialize = A         , doc ='Parámetro A') 
    model.B          = Param(model.G, initialize = B         , doc ='Parámetro B') 
    model.BH         = Param(model.H, initialize = BH        , doc ='Parámetro B generadores Hidro') 
    model.CA         = Param(model.G, initialize = CA        , doc ='Costo fijo de arranque de generadores termo')
    model.CP         = Param(model.G, initialize = CP        , doc ='Costo fijo de parada de generadores termo')
    model.PGmax      = Param(model.G, initialize = PGmax     , doc ='Limite maximo de potencia térmica generada')
    model.PGmin      = Param(model.G, initialize = PGmin     , doc ='Limite minimo de potencia térmica generada') 
    model.PGsubida   = Param(model.G, initialize = PGsubida  , doc ='Rampa de subida para generadores termos')
    model.PGbajada   = Param(model.G, initialize = PGbajada  , doc ='Rampa de bajada para generadores termos')
    model.PGarranque = Param(model.G, initialize = PGarranque, doc ='Rampa de arranque para generadores termos')
    model.PGparada   = Param(model.G, initialize = PGparada  , doc ='Rampa de parada para generadores termos')
    model.const      = Param(model.H, initialize = const     , doc ='Constante de eficiencia en MWs/m^3')
    model.r          = Param(model.H, initialize = r         , doc ='Influjos naturales en m^3/s', within=Any) 
    model.Vo         = Param(model.H, initialize = Vo        , doc ='volumen inicial del enbalse en km^3')
    model.PHmax      = Param(model.H, initialize = PHmax     , doc ='Limite maxima de potencia hidro generada')
    model.PHmin      = Param(model.H, initialize = PHmin     , doc ='Limite minima de potencia hidro generada')
    model.PHsubida   = Param(model.H, initialize = PHsubida  , doc ='Rampa de subida para generadores hidros')
    model.PHbajada   = Param(model.H, initialize = PHsubida  , doc ='Rampa de bajada para generadores hidros')
    model.Vmax       = Param(model.H, initialize = Vmax      , doc ='Volumen maximo del embalse') 
    model.Vmin       = Param(model.H, initialize = Vmin      , doc ='Volumen minimo del embalse')
    model.Qmax       = Param(model.H, initialize = Qmax      , doc ='Caudal maximo turbinado')
    model.Qmin       = Param(model.H, initialize = Qmin      , doc ='Caudal minimo turbinado')
    model.Upstream   = Param(model.H, initialize = Upstream  , doc ='Hidros aguas arriba', within=Any)
    
    if metodo == 'B':
        model.Dem        = Param(model.T, model.N, initialize = tablademanda, doc ='Demanda por periodo en MW')
        model.Plinemax   = Param(model.L, initialize = Prama_max , doc ='Limite maximo de potencia en las lineas de transmisión MW')
        model.S          = Param(model.N, model.N, initialize = Bbus, doc ='Valores de susceptancla de las lineas de transmisión')
        BA = model.Base  = Param(initialize = 100, doc ='Potencia base MVA')
    else :
         model.Dem        = Param(model.T, initialize = tablademanda, doc = 'Demanda por periodo')
    
    # VARIABLES
    model.P  = Var(model.G, model.T, within=NonNegativeReals, doc ='Potencia termoelectrica generada') 
    def limitesPH(model,h,t):
        return (model.PHmin[h],model.PHmax[h])
    model.PH = Var(model.H, model.T, within=NonNegativeReals, doc ='Potencia hidroelectrica generada',bounds=limitesPH)
    model.U  = Var(model.G, model.T, within=Binary,           doc ='Variable binaria de acoplamiento')
#     Activar el lazo for anidado para tener siempre en línea las unidades
    for g in model.G:
        for t in model.T:
            model.U[g,t].fix(1)
    model.Y  = Var(model.G,	model.T, within=Binary,           doc ='Variable binaria de arranque')
    model.W  = Var(model.G,	model.T, within=Binary,           doc ='Variable binaria de parada')
    def limitesQ(model,h,t):
        return (model.Qmin[h],model.Qmax[h])
    model.Q  = Var(model.H, model.T, within=NonNegativeReals, doc ='Caudal turbinado',bounds=limitesQ) 
    for h in model.H:
        model.Q[h,0].fix(0)
    def limitesVol(model,h,t):
        return (1000000*model.Vmin[h],1000000*model.Vmax[h])
    model.Vol = Var(model.H, model.T, within=NonNegativeReals, doc = 'Volumen del embalse al final de la hora T m^3',bounds=limitesVol)
    for h in model.H:
        model.Vol[h,0].fix(1000000*model.Vo[h])
    
    if metodo == 'B':
        model.D = Var(model.N, model.T, bounds=(-3.1416,3.1416), doc ='Angulo de voltaje en las barras rad') 
        for t in model.T:
            model.D[Slack, t].fix(0)
             
    # RESTRICCIONES  
    def pottermomax(model, g, t):
        return model.P[g,t] <= model.PGmax[g]*model.U[g,t]
    model.PGmaxima = Constraint(model.G, model.T, rule=pottermomax, doc = 'Maxima potencia termica generada')
    
    def pottermomin(model, g, t):
        return model.P[g,t] >= model.PGmin[g]*model.U[g,t]
    model.PGminima = Constraint(model.G, model.T, rule=pottermomin, doc = 'Minima potencia termica generada')
    
    def potterarranquesubida(model, g, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.P[g,t]- model.P[g,t-1] <= model.PGsubida[g]*model.U[g,t-1]+model.PGarranque[g]*model.Y[g,t]
    model.PGArranSub = Constraint(model.G, model.T, rule=potterarranquesubida, doc ='Rampa de arranque y subida')

    def potterparadabajada(model, g, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.P[g,t-1]-model.P[g,t] <= model.PGbajada[g]*model.U[g,t]+model.PGparada[g]*model.W[g,t]
    model.PGParadBaj = Constraint(model.G, model.T, rule=potterparadabajada, doc ='Rampa de parada y bajada') 

    def logicabinaria1(model, g, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.U[g,t]-model.U[g,t-1] == model.Y[g,t]-model.W[g,t]
    model.logicbin1 = Constraint(model.G, model.T, rule=logicabinaria1, doc ='Primera restricción binaria') 

    def logicabinaria2(model, g, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.Y[g,t]+model.W[g,t] <= 1.0
    model.logicbin2 = Constraint(model.G, model.T, rule=logicabinaria2, doc ='Segunda restricción binaria')

    def pothidrosubida(model, h, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.PH[h,t]-model.PH[h,t-1] <= model.PHsubida[h]
    model.PHSub = Constraint(model.H, model.T, rule=pothidrosubida, doc ='Rampa de arranque y subida')
    
    def pothidrobajada(model, h, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.PH[h,t-1]-model.PH[h,t] <= model.PHbajada[h]
    model.PHBaj = Constraint(model.H, model.T, rule=pothidrobajada, doc ='Rampa de parada y bajada')
    
    def caudaltotal(model, h, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.Q[h,t] == model.PH[h,t]/model.const[h]
    model.Caudal = Constraint(model.H, model.T, rule=caudaltotal, doc = 'Ecuacion de caudal turbinado')
    
    def voltotalemb(model, h, t):
        if t == 0:
            return Constraint.Skip
        else:
            return model.Vol[h,t] == model.Vol[h,t-1] + (3600*model.r[h][t]) - (model.Q[h,t])+sum(model.Q[ha,t] for ha in Upstream[h])
    model.Volumen = Constraint(model.H, model.T, rule=voltotalemb, doc = 'Volumen total del embalse en el periodo T')
    
    if metodo == 'A':
       def sumpot(model, t):
           if t == 0:
               return Constraint.Skip
           return sum(model.P[g,t] for g in model.G) + sum(model.PH[h,t] for h in model.H) == model.Dem[t] 
       model.Carga = Constraint(model.T, rule=sumpot, doc = 'Potencia generada igual a la demanda') 
    
    else:
        def sumpot(model, n, t):
            if t == 0:
                return Constraint.Skip
            else:
                return (sum(model.P[g,t] for g in model.G if (g,n) in model.MAPG)/BA) + (sum(model.PH[h,t] for h in model.H if(h,n) in model.MAPH)/BA) - (model.Dem[t,n]/BA) == sum( ((model.D[n,t]-model.D[np,t])*model.S[n,np]) for np in model.NP) 
        model.Carga = Constraint(model.N, model.T, rule=sumpot, doc ='Potencia generada igual a la demanda MW')
        
        def potij(model,l,t):
            if t == 0:
                return Constraint.Skip
            else:
                for n in model.N :
                    for np in model.NP:
                        if (l,n,np) in model.MAPL:
                            return (((model.D[n,t]-model.D[np,t])*model.S[n,np]) )*BA <= (model.Plinemax[l])
        model.Poij = Constraint(model.L, model.T, rule=potij, doc = 'Potencia en la rama ij MW')
    
    # FUNCION OBJETIVO
    def objetivo(model):
        return sum((model.A[g]*model.U[g,t])+(model.B[g]*model.P[g,t])+(model.CA[g]*model.Y[g,t]) + (model.CP[g]*model.W[g,t])for g in model.G for t in model.T if t != 0)+sum((model.BH[h]*model.PH[h,t])for h in model.H for t in model.T if t != 0)
    model.Obj = Objective(rule = objetivo, sense = minimize, doc ='Función objetivo')
    
    model.rc = Suffix(direction=Suffix.IMPORT)
    
#%%   
    def pyomo_postprocess(options=None, instance=None, results=None): 
        model.Obj.display()
        model.P.display()
        model.PH.display()
        if metodo == 'B':
            model.Poij.display()
#         model.dual.display()
#         model.Q.display() 
#         model.Vol.display()
#         model.U.display()
#         model.Y.display()
#         model.W.display()
    #%%
   
    opt = SolverFactory(solver)
    try:
        results= opt.solve(model)
    except:
        results = log_infeasible_constraints(model)
        print(results)
    
    for i in model.U:
        if model.U[i].value == 0:
            model.U[i].fix(0)
        else:
            model.U[i].fix(1)
        if model.Y[i].value == 0:
            model.Y[i].fix(0)
        else:
            model.Y[i].fix(1)
        if model.W[i].value == 0:
            model.W[i].fix(0)
        else:
            model.W[i].fix(1)
    try:
        results= opt.solve(model)
#        pyomo_postprocess()
    except:
        results = log_infeasible_constraints(model)
        print(results)
    #%%
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('Se halló una solución óptima y factible')
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print('Solución inviable ')
        sys.exit(1)
    else:
        print ('Solver Status: '),  result.solver.status
        sys.exit(1)
    #%%
    PHFija,GenGraph,PGini,in_service = [],[],[],[]
    for i in model.PH:
        GenGraph.append(model.PH[i].value)
        PHFija.append(model.PH[i].value)
    PHFija = [PHFija[i:i+len(Tabla_Demanda)+1] for i in range(0,len(PHFija),len(Tabla_Demanda)+1)]
    PHFija = [i[1:] for i in PHFija]

    for i in model.P:
        GenGraph.append(model.P[i].value)
        PGini.append(model.P[i].value)
    PGini = [PGini[i:i+len(Tabla_Demanda)+1] for i in range(0,len(PGini),len(Tabla_Demanda)+1)]
    PGini = [i[1:] for i in PGini]
    data = [GenGraph[n:n+len(Tabla_Demanda)+1] for n in range(0, len(GenGraph), len(Tabla_Demanda)+1)]
    
    for i in model.U:
        in_service.append(model.U[i].value)
    in_service = [in_service[n:n+len(Tabla_Demanda)+1] for n in range(0, len(in_service), len(Tabla_Demanda)+1)]
    in_service = [i[1:] for i in in_service] 
#     model.pprint()
    
    if metodo == 'B':
        LaM = [model.dual[model.Carga[(Slack,key)]]/BA for key in range(1,25)]
    else:
        LaM = [model.dual[model.Carga[key]] for key in model.Carga.keys()]
    RC_dict = {index:model.rc[c[index]] for c in model.component_objects(Var, active=True) if c == model.P or c== model.PH for index in c}       
     #%%
    return model.Obj.expr(),model,results,PHFija,GenGraph,PGini,data,in_service,LaM,RC_dict,pyomo_postprocess