function [] = writeSTO_Updated(data,header,filePath,fileName,version,inDegrees) ;
% Function updated by Janelle K (Nov 2022): updated header entries to be
% consistent with OAGR file types

% Columnwise data, header in a cell
% version: version number (int)
% inDegrees: 1 = yes, 0 = no

    fid = fopen([filePath '\' fileName '.sto'],'w') ;
% Write the header
    if contains(fileName, 'ik')
         fprintf(fid,'%s\n','Coordinates');
    else
        fprintf(fid,'%s\n',fileName);
    end
    fprintf(fid,'%s\n',['version=' num2str(version)]);
    fprintf(fid,'%s\n',['nRows=' num2str(size(data,1))]);
    fprintf(fid,'%s\n',['nColumns=' num2str(length(header))]);
    if inDegrees
        fprintf(fid,'%s\n','inDegrees=yes');
    else
        fprintf(fid,'%s\n','inDegrees=no');
    end
    fprintf(fid,'%s\n',''); % skip a line
    fprintf(fid,'%s\n','Units are S.I.units (second, meters, Newtons, ...');
    if inDegrees
        fprintf(fid,'%s\n','Angles are in degrees.');
    else
        fprintf(fid,'%s\n','Angles are in radians.');
    end
    fprintf(fid,'%s\n',''); % skip a line
    fprintf(fid,'%s\n','endheader');

    % Print headers
    for i = 1:length(header)-1
        fprintf(fid,'%s\t',header{i}) ;
    end
    fprintf(fid,'%s\n',header{end}) ;
    
    % Write the data
    for j=1:size(data,1)
        fprintf(fid,'%.6f\t',data(j,1:end-1));
        fprintf(fid,'%.6f',data(j,end));
        fprintf(fid,'\n');
    end
    
    fclose all ;
    
    disp(['Successfully wrote ' filePath '\' fileName '.sto to file.']) ;