import os
import pandas as pd
from datetime import timedelta

def report(result_dir,model,list_of_dataframes, metodo,runtime,iteracion):
    """Write result summary to a spreadsheet file

    Args:
        - 
        - filename: Excel spreadsheet filename, will be overwritten if exists;
        - report_tuples: (optional) list of (sit, com) tuples for which to
          create detailed timeseries sheets;
        - report_sites_name: (optional) dict of names for created timeseries
          sheets
    """

    if metodo == 'A':
        output_file = "Solucion Método A"
        
    else:
        output_file = "Solucion Método B"
            
    output_path = os.path.join(result_dir, output_file +'.txt')   

    with open(output_path, 'w') as f:
        f.write("Optimización Hidrotérmica Final!\n\n")
        f.write("Iteración: {} \n\n".format(iteracion))
        f.write('Tiempo de Ejecución:{} \n\n'.format(timedelta(seconds=runtime)))
        model.Obj.display(ostream=f)
        model.P.display(ostream=f)
        model.PH.display(ostream=f)
        if metodo == 'B':
            model.Poij.display(ostream=f)
            model.D.display(ostream=f)
        model.Q.display(ostream=f) 
        model.Vol.display(ostream=f)
        model.U.display(ostream=f)
        model.Y.display(ostream=f)
        model.W.display(ostream=f)
    
    # create spreadsheet writer object
    
    
    
       # write constants to spreadsheet
    for i in range(len(list_of_dataframes)):
        if i == 0:
            output_file = "Respuesta Barras"
            sheet_name  = 'res_bus_h_'
        elif i == 1:
            output_file = "Respuesta Generadores"
            sheet_name  = 'res_gen_h_'
        elif i == 2:
            output_file = "Respuesta líneas"
            sheet_name  = 'res_line_h_'
        else:
            output_file = "Respuesta Transformadores"
            sheet_name  = 'res_trafo_h_'
        output_path = os.path.join(result_dir, output_file + '.xls') 
        
        with pd.ExcelWriter(output_path) as writer:
            for j in range(len(list_of_dataframes[i])):
                 nombre_hoja = sheet_name + str(j+1)
                 if i == 1:
                     list_of_dataframes[i][j].loc[[0,1,2,3,4]].to_excel(writer, nombre_hoja)
                 else:
                    list_of_dataframes[i][j].to_excel(writer, nombre_hoja)
        