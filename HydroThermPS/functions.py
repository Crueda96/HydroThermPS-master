import os
import pandapower as pp
from datetime import datetime
from .report import *
from .plot import *

#%%
def prepare_result_directory(result_name):
    """ create a time stamped directory within the result folder.

    Args:
        result_name: user specified result name

    Returns:
        a subfolder in the result folder 
    
    """
    # timestamp for result directory
    now = datetime.now().strftime('%Y%m%dT%H%M')

    # create result directory if not existent
    result_dir = os.path.join('Output_files', '{}-{}'.format(result_name, now))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir

#%% Función Distribución de Pérdidas Incrementales

def DistPerdidas (net, load_caso_base):
    net.load = load_caso_base.copy() 
    for i in range(len(net.bus)):
        if not net.load.bus.isin([i]).any() :
            nn='Load Loss '+ str(i+1)
            pp.create_load(net,bus=i,p_mw=0,name=nn)
    net.load = net.load.sort_values('bus',ascending=True)
    net.load = net.load.reset_index(drop=True)       
    for i in range(len(net.bus)):    
        aux = 0
        b = net.line.from_bus[net.line.from_bus == i].index.tolist()
        c = net.line.to_bus[net.line.to_bus == i].index.tolist()
        d = net.trafo.hv_bus[net.trafo.hv_bus == i].index.tolist()
        e = net.trafo.lv_bus[net.trafo.lv_bus == i].index.tolist()
       
        for j in b:
            aux = aux +  (net.res_line.pl_mw[j])/2 
        for j in c:
            aux = aux + (net.res_line.pl_mw[j])/2    
        for j in d:
            aux = aux + (net.res_trafo.pl_mw[j])/2
        for j in e:
            aux = aux + (net.res_trafo.pl_mw[j])/2
        
        net.load.p_mw.loc[i]=net.load.p_mw[i]+aux


#%% Función Error

def ErrorLazo(V,V_i,cont):
    if V_i == 0:
        err = 100
    else:
        err = abs(V-V_i)/V_i
    return(err,cont)
    
#%% Funcion Iterativa Interior

def LazoInt(net,load_caso_base,GenReales,metodo):
    # Inicio Lazo Interior [t : 1 h]
    index     = net.gen.index
    condicion = net.gen["type"] == "Ficticio"
    GenFict   = index[condicion]
    if metodo == 'A':
        err1 ,PL_i,cont1 = 100, 0, 0
        while err1 >= 10e-6 and cont1<=10:
            try:
                pp.runpm_ac_opf(net)                             # Optimización FOP AC
                net.gen.p_mw[GenReales] = net.res_gen.p_mw[GenReales]
            except:
                print("OPF AC No converge")
                break
                    
            DistPerdidas(net, load_caso_base)         # Distribución de Pérdidas en Barras
            Load_loss = net.load.p_mw.sum()           # Cálculo de Nueva Demanda con pérdidas
            PL = Load_loss-load_caso_base.p_mw.sum()  # Cálculo de Pérdidas Totales del Sistema
            # Criterio de convergencia     
            err1,cont1 = ErrorLazo(PL,PL_i,cont1)
            cont1 += 1
            PL_i  = PL
             
#        df1      =  net.res_gen[['vm_pu','p_mw']].loc[[0,1,2,3,4]]
        if cont1 >=10 or err1>=10e-6:
            print('Iteración: ',cont1)
            print('Error    : ', err1 , '%\n')
            print('============================\n¡NO CONVERGE!\n============================')
    #    else:
    #        print('¡CONVERGIO!')
    #        print('Iteración: ',cont1)
    #        print('Error    : ', err1 , '%\n')
    #        print('Generacion Óptima:\n',df1)
    #        print('Demanda Total:',net.load.p_mw.sum())
    #        print('Demanda Inicial:', load_caso_base.p_mw.sum())

    else : 
        try: 
            pp.runpm_ac_opf(net)                  # Cálculo de FOP AC y Cálculo de Pérdidas Incrementales
        except:
            print("**¡¡FOP AC no converge!!**")
                              
        DistPerdidas (net, load_caso_base)                           # Distribución de Pérdidas en Barras
        Load_loss = net.load.p_mw.sum()           # Cálculo de Nueva Demanda con pérdidas
        PL = Load_loss-load_caso_base.p_mw.sum()  # Cálculo de Pérdidas Totales del Sistema 
        PL_i  = PL
      
#        df1      = net.res_gen[['vm_pu','p_mw']].loc[[0,1,2,3,4]]
        

    #     df1      = net.res_gen[['vm_pu','p_mw']]
#        print('Generacion Óptima:\n',df1)
#        print('Demanda Inicial:', load_caso_base.p_mw.sum())
#        print('Demanda Total:',net.load.p_mw.sum())
#        print('Perdidas del Sistema:', net.res_line.pl_mw.sum()+net.res_trafo.pl_mw.sum())
    
    for idx,i in enumerate(net.res_gen.p_mw[GenFict].ge(0),start=GenFict[0]): 
        if i != False:
            print ("Generador {} Ficticio Activado!".format(net.gen.name[idx]))
        
#%%# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
