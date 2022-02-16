import pandas as pd 
import geopandas as gp
import numpy as np 
import pylab as pl 
from struct import pack, unpack
import io
import gdal
from osgeo import ogr
import osgeo
#from wmf import wmf 
pd.options.mode.chained_assignment = None

def read_raster(path_map,isDEMorDIR=False,dxp=None, noDataP = None,isDIR = False,DIRformat = 'r.watershed'):
    'Funcion: read_map\n'\
    'Descripcion: Lee un mapa raster soportado por GDAL.\n'\
    'Parametros Obligatorios:.\n'\
    '   -path_map: path donde se encuentra el mapa.\n'\
    'Parametros Opcionales:.\n'\
    '   -isDEMorDIR: Pasa las propiedades de los mapas al modulo cuencas \n'\
    '       escrito en fortran \n'\
    '   -dxp: tamano plano del mapa\n'\
    '   -noDataP: Valor para datos nulos en el mapa (-9999)\n'\
    '   -DIRformat: donde se ha conseguido el mapa dir (r.watershed) \n'\
    '       - r.watershed: mapa de direcciones obtenido por la funcion de GRASS\n'\
    '       - opentopo: mapa de direcciones de http://www.opentopography.org/\n'\
    '   -isDIR: (FALSE) es este un mapa de direcciones\n'\
    'Retorno:.\n'\
    '   Si no es DEM o DIR retorna todas las propieades del elemento en un vector.\n'\
    '       En el siguiente orden: ncols,nrows,xll,yll,dx,nodata.\n'\
    '   Si es DEM o DIR le pasa las propieades a cuencas para el posterior trazado.\n'\
    '       de cuencas y link_ids.\n' \
    #Abre el mapa
    direction=gdal.Open(path_map)
    #Projection
    proj = osgeo.osr.SpatialReference(wkt=direction.GetProjection())
    EPSG_code = proj.GetAttrValue('AUTHORITY',1)
    #lee la informacion del mapa
    ncols=direction.RasterXSize
    nrows=direction.RasterYSize
    banda=direction.GetRasterBand(1)
    noData=banda.GetNoDataValue()
    geoT=direction.GetGeoTransform()
    dx=geoT[1]
    dy = np.abs(geoT[-1])
    xll=geoT[0]; yll=geoT[3]-nrows*dy
    #lee el mapa
    Mapa=direction.ReadAsArray()
    direction.FlushCache()
    del direction
    return Mapa.T.astype(float),[ncols,nrows,xll,yll,dx,dy,noData],EPSG_code

def save_array2raster(Array, ArrayProp, path, EPSG = 4326, Format = 'GTiff'):
    dst_filename = path
    #Formato de condiciones del mapa
    x_pixels = Array.shape[0]  # number of pixels in x
    y_pixels = Array.shape[1]  # number of pixels in y
    PIXEL_SIZE_x = ArrayProp[4]  # size of the pixel... 
    PIXEL_SIZE_y = ArrayProp[5]  # size of the pixel...
    x_min = ArrayProp[2]
    y_max = ArrayProp[3] + ArrayProp[5] * ArrayProp[1] # x_min & y_max are like the "top left" corner.
    driver = gdal.GetDriverByName(Format)
    #Para encontrar el formato de GDAL
    NP2GDAL_CONVERSION = {
      "uint8": 1,
      "int8": 1,
      "uint16": 2,
      "int16": 3,
      "uint32": 4,
      "int32": 5,
      "float32": 6,
      "float64": 7,
      "complex64": 10,
      "complex128": 11,
    }
    gdaltype = NP2GDAL_CONVERSION[Array.dtype.name]
    # Crea el driver
    dataset = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1,
        gdaltype,)
    #coloca la referencia espacial
    dataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE_x,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE_y))
    #coloca la proyeccion a partir de un EPSG
    proj = osgeo.osr.SpatialReference()
    texto = 'EPSG:' + str(EPSG)
    proj.SetWellKnownGeogCS( texto )
    dataset.SetProjection(proj.ExportToWkt())
    #Coloca el nodata
    band = dataset.GetRasterBand(1)
    if ArrayProp[-1] is None:
        band.SetNoDataValue(-9999)
    else:
        band.SetNoDataValue(int(ArrayProp[-1]))
    #Guarda el mapa
    dataset.GetRasterBand(1).WriteArray(Array.T)
    dataset.FlushCache()

def rainfall_raster_ranks(path_rain_frame, path_ranks):
    # Reads a raster of the rainfall fields and creates a raster with the ranks 
    m, p, epsg = read_raster(path_rain_frame)
    rank = np.arange(1,m.size+1)
    rank = rank.reshape(m.shape)
    save_array2raster(rank , p, path_ranks+'.tif', EPSG=int(epsg))
    # Creates a ranks polygon based on the raster ranks.
    src_ds = gdal.Open(path_ranks+'.tif')
    srcband = src_ds.GetRasterBand(1)
    #Create output datasource
    spatialReference = osgeo.osr.SpatialReference()
    spatialReference.ImportFromEPSG(int(epsg))
    dst_layername = path_ranks
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( dst_layername + ".shp" )
    dst_layer = dst_ds.CreateLayer(dst_layername, spatialReference )
    gdal.Polygonize( srcband, None, dst_layer, -1, [], callback=None )
    dst_ds.Destroy()

def saveBin(lid, lid_vals, count, fn):
    io_buffer_size = 4+4*100000
    if count > 0:
        lid = (lid[lid_vals > 1])
        lid_vals = (lid_vals[lid_vals > 1])
    fh = io.open(fn, 'wb', io_buffer_size)
    fh.write(pack('<I', count))
    for vals in zip(lid, lid_vals):
        fh.write(pack('<If', *vals))
    fh.close()

class network:
    
    def __init__(self, net_path, hills_path = None, hills_epsg = 2163):
        '''Defines the network class that contains all the requirements to set up a project for
        hlm'''
        if type(net_path) is str:
            #Defines the initial partameters for the network
            self.network = gp.read_file(net_path)        
            self.network['link'] = self.network['LINKNO']
            self.network.set_index('LINKNO', inplace=True)            
            self.network_centroids = None
            self.network_ranks = None
            #computes the area for each hillslope
            if hills_path is not None:
                self.hills = gp.read_file(hills_path)
                self.hills.rename(columns={'DN':'link'}, inplace = True)
                self.hills.set_index('link', inplace = True)
                self.hills.to_crs(epsg = hills_epsg, inplace = True)
                idx = self.hills.index.intersection(self.network.index)
                self.network['area'] = self.hills.loc[idx].geometry.area/1e6
                print('Area of each hillslope computed from the hills shapefile')
            else:
                self.hills = None
        elif type(net_path) is gp.geodataframe.GeoDataFrame:
            self.network = net_path.copy()
            if type(hills_path) is gp.geodataframe.GeoDataFrame:
                self.hills = hills_path.copy()
            
            
    
    def network2points(self):
        '''Converts the network elements to centroids, ideal to get the 
        rainfall ranks references'''
        x =[]
        y = []
        for link in self.network.index:
            geo = self.network.loc[link, 'geometry']
            x.append(geo.centroid.x)
            y.append(geo.centroid.y)
        net_centroids = gp.GeoDataFrame(self.network[['link','strmOrder']], geometry = gp.points_from_xy(x, y),
                                crs = self.network.crs)
        self.network_centroids = net_centroids
        print('Centroids had been saved under self.network_centroids')
        #return net_centroids
    
    def get_rainfall_lookup(self, path_rain_ranks):
        '''Generates the lookup table between the links and a rainfall that is going to be used
        the rain ranks must be the one obtained with *rainfall_raster_ranks*. By now this operation
        is done one to one.'''
        # Reads the rainfall ranks and project it
        rain_ranks = gp.read_file(path_rain_ranks)
        rain_ranks = rain_ranks.to_crs(self.network.crs)
        print('1. rain ranks readed and projected to the current crs')
        # Checks if centroids are already defined
        if self.network_centroids is None:
            print('2. Network points not defined, defining them...')
            self.network2points()
        print('3. Network points defined')    
        # Performs the spatial join
        points_ranked = gp.sjoin(self.network_centroids, rain_ranks, how = 'left', op = 'within')
        self.rain_ranks = points_ranked
        print('4. ranks obtained results stored in self.rain_ranks')
    
    def rain2links(self, rain, path_rain = None):
        '''Converts a grid (tif) file of rainfall to the shape of the network 
        using the lookup table obtained by *get_rainfall_lookup*'''
        if rain is None:
            if path_rain is not None:
                #Read and transform rainfall to its ranks    
                rain, p, ep = read_raster(path_rain)
                rain = rain.T
                rain = rain.reshape(rain.size)
            else:
                print('Error: No rain variable, no path to rain variable')
        else:
            rain = rain.reshape(rain.size)
        #Put the rinfall in links 
        self.rain_ranks['rain'] = 0
        self.rain_ranks['rain'] = rain[self.rain_ranks['FID']]
        # Return the links and the rainfall 
        return self.rain_ranks['rain']
    
    def write_rvr(self, path, sub_net = None):
        '''Writes and rvr file based on a network extracted from the base network'''
        #Selects the subnet if it is avaiable
        if sub_net is not None:
            net_elem = sub_net
        else:
            net_elem = self.network
        #Writes the rvr file for HLM
        with open(path,'w',newline='\n') as f:
            f.write('%d\n' % net_elem.shape[0])
            f.write('\n')
            for link in net_elem.index:
                f.write('%d\n' % link)
                if net_elem.loc[link,'USLINKNO1'] == -1:
                    f.write('0\n')
                else:
                    f.write('2 %d %d\n' % (net_elem.loc[link,'USLINKNO1'], net_elem.loc[link,'USLINKNO2']))
                f.write('\n')
            f.close()
        
    def get_subnet(self, link, get_divisory=False):
        '''Allows to define a new network inside of the base network'''
        lista = [link]
        count = 0
        while count < len(lista) or count > self.network.shape[0]:
            link = lista[count]
            if self.network.loc[link, 'USLINKNO1'] != -1:
                lista.append(self.network.loc[link, 'USLINKNO1'])
                lista.append(self.network.loc[link, 'USLINKNO2'])
            count += 1
        idx_links = self.network.index.intersection(lista)
        if self.hills is not None:
            idx_hills = self.hills.index.intersection(lista)
            new_net = network(self.network.loc[idx_links], self.hills.loc[idx_hills])
            if get_divisory:
                new_net.hills['loc_id'] = link
                new_net.divisory = new_net.hills.dissolve(by='loc_id')                
        else:
            new_net = network(self.network.loc[idx_links])
        return new_net
    
    def get_prm(self):
        for_prm = self.network[['DSContArea','Length','area']]
        for_prm['DSContArea'] = for_prm['DSContArea'] / 1e6
        for_prm.shape[0] == self.network.shape[0]

        for_prm.loc[for_prm['Length'] == 0, 'Length'] = 1
        for_prm.loc[for_prm['area'] == 0, 'area'] = 1/1e4
        for_prm.loc[np.isnan(for_prm['area']), 'area'] = 1/1e4
        for_prm['Length'] = for_prm['Length'] / 1000
        self.prm = for_prm
        
    def set_prm_for_model(self, model = 608):
        if model == 608 or model == 609:
            attr = {'vh':0.02,'a_r':1.67,'a':3.2e-6,'b':17,'c':5.4e-7,'d':32,
                'k3':2.045e-6,'ki_fac':0.07,'TopDepth':0.1,'NoFlow':1.48,'Td':999,
                'Beta':1.67,'lambda1':0.4,'lambda2':-0.1,'vo':0.435}
            self.prm_format = {'DSContArea':'%.3f','Length':'%.3f','area':'%.5f',
                'vh':'%.4f','a_r':'%.4f','a':'%.2e','b':'%.1f','c':'%.2e','d':'%.1f',
                    'k3':'%.2e','ki_fac':'%.3f','TopDepth':'%.3f','NoFlow':'%.3f','Td':'%.2f',
                    'Beta':'%.3f','lambda1':'%.3f','lambda2':'%.2f','vo':'%.3f'}
            self.prm = self.prm.assign(**attr)
        elif model == 254 or model == 253 or model == 256:
            self.prm_format = {'DSContArea':'%.3f','Length':'%.3f','area':'%.5f'}
        
    def write_prm(self, path):
        with open(path,'w',newline='\n') as f:
            f.write('%d\n\n' % self.prm.shape[0])
            for link in self.prm.index:
                f.write('%d\n' % link)
                for c,k in zip(self.prm.loc[link],self.prm_format.keys()):
                    fm = self.prm_format[k]+' '
                    f.write(fm % c)
                f.write('\n\n')
                
    def write_Global(self, path2global, model_uid = 604,
        date1 = None, date2 = None, rvrFile = None, rvrType = 0, rvrLink = 0, prmFile = None, prmType = 0, initialFile = None,
        initialType = 1,rainType = 5, rainPath = None, evpFile = 'evap.mon', datResults = None,
        nComponents = 1, Components = [0], controlFile = None, baseGlobal = None, noWarning = False, snapType = 0,
        snapPath = '', snapTime = '', evpFromSysPath = False):
        '''Creates a global file for the current project.
            - model_uid: is the number of hte model goes from 601 to 604.
            - date1 and date2: initial date and end date
            - rvrFile: path to rvr file.
            - rvrType: 0: .rvr file, 1: databse .dbc file.
            - rvrLink: 0: all the domain, N: number of the linkid.
            - prmFile: path to prm file.
            - prmType: 0: .prm file, 1: databse .dbc file.
            - initialFile: path to file with initial conditions.
            - initialType: type of initial file:
                - 0: ini, 1: uini, 2: rec, 3: .dbc
            - rainType: number inficating the type of the rain to be used.
                - 1: plain text with rainfall data for each link.
                - 3: Database.
                - 4: Uniform storm file: .ustr
                - 5: Binary data with the unix time
            - rainPath: path to the folder containning the binary files of the rain.
                or path to the file with the dabase
            - evpFile: path to the file with the values of the evp.
            - datResults: File where .dat files will be written.
            - nComponents: Number of results to put in the .dat file.
            - Components: Number of each component to write: [0,1,2,...,N]
            - controlFile: File with the number of the links to write.
            - baseGlobal: give the option to use a base global that is not the default
            - snapType: type of snapshot to make with the model:
                - 0: no snapshot, 1: .rec file, 2: to database, 3: to hdf5, 4:
                    recurrent hdf5
            - snapPath: path to the snapshot.
            - snapTime: time interval between snapshots (min)
            - evpFromSysPath: add the path of the system to the evp file.'''
        #Open the base global file and creates tyhe template
        if baseGlobal is not None:
            f = open(baseGlobal, 'r')
            L = f.readlines()
            f.close()
        else:
            L = Globals['60X']
        t = []
        for i in L:
            t += i
        Base = Template(''.join(t))
        # Databse rainfall 
        if rainType == 3 and rainPath is None:
            rainPath = '/Dedicated/IFC/model_eval/forcing_rain51_5435_s4.dbc'
        if rvrType == 1 and rvrFile is None:
            rvrFile = '/Dedicated/IFC/model_eval/topo51.dbc'
        #Chang  the evp path 
        if evpFromSysPath:
            evpFile = Path + evpFile
        # Creates the default Dictionary.
        Default = {
            'model_uid' : model_uid,
            'date1': date1,
            'date2': date2,
            'rvrFile': rvrFile,
            'rvrType': str(rvrType),
            'rvrLink': str(rvrLink),
            'prmFile': prmFile,
            'prmType': str(prmType),
            'initialFile': initialFile,
            'initialType': initialType,
            'rainType': str(rainType),
            'rainPath': rainPath,
            'evpFile': evpFile,
            'datResults': datResults,
            'controlFile': controlFile,
            'snapType': str(snapType),
            'snapPath': snapPath,
            'snapTime': str(snapTime),
            'nComp': str(nComponents)
        }
        if date1 is not None:
            Default.update({'unix1': aux.__datetime2unix__(Default['date1'])})
        else:
            Default.update({'unix1': '$'+'unix1'})
        if date2 is not None:
            Default.update({'unix2': aux.__datetime2unix__(Default['date2'])})
        else:
            Default.update({'unix2': '$'+'unix2'})
        #Update the list of components to write
        for n, c in enumerate(Components):
            Default.update({'Comp'+str(n): 'State'+str(c)})
        if nComponents <= 9:
            for c in range(9-nComponents):
                Default.update({'Comp'+str(8-c): 'XXXXX'})
        #Check for parameters left undefined
        D = {}
        for k in Default.keys():
            if Default[k] is not None:
                D.update({k: Default[k]})
            else:
                if noWarning:
                    print('Warning: parameter ' + k +' left undefined model wont run')
                D.update({k: '$'+k})
        #Update parameter on the base and write global 
        f = open(path2global,'w', newline='\n')
        f.writelines(Base.substitute(D))
        f.close()
        #Erase unused print components
        f = open(path2global,'r')
        L = f.readlines()
        f.close()
        flag = True
        while flag:
            try:
                L.remove('XXXXX\n')
            except:
                flag = False
        f = open(path2global,'w', newline='\n')
        f.writelines(L)
        f.close()


    def write_runfile(self, path, process, jobName = 'job',nCores = 56, nSplit = 1, queue = 'IFC'):
        '''Writes the .sh file that runs the model
        Parameters:
            - path: path where the run file is stored.
            - process: dictionary with the parameters for each process to be launch:
                eg: proc = {'Global1.gbl':{'nproc': 12, 'secondplane': True}}
            - ncores: Number of cores.
            - nsplit: Total number of cores for each group.
            - queue: name of the argon queue to run the process'''
        #Define the size of the group of cores
        if nCores%nSplit == 0:
            Groups = int(nCores / nSplit)
        else:
            Groups = int(nCores / 2)
        #Define the header text.
        L = ['#!/bin/sh\n#$ -N '+jobName+'\n#$ -j y\n#$ -cwd\n#$ -pe smp '+str(nCores)+'\n####$ -l mf=16G\n#$ -q '+str(queue)+'\n\n/bin/echo Running on host: `hostname`.\n/bin/echo In directory: `pwd`\n/bin/echo Starting on: `date`\n']
        f = open(path,'w',  newline='\n')
        f.write(L[0])
        f.write('\n')

        for k in process.keys():
            secondplane = ' \n'
            if process[k]['secondplane']:
                secondplane = ' &\n'
            if process[k]['nproc'] > nCores:
                process[k]['nproc'] = nCores        
            f.write('mpirun -np '+str(process[k]['nproc'])+' /Users/nicolas/Tiles/dist/bin/asynch '+k+secondplane)
        f.close()

        
        

