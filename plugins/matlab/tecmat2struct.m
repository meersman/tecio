function zones = tecmat2struct(d)
% Reshape a loaded tec2mat .mat struct into a struct array indexed by zone,
% with dataset variables accessible by their real Tecplot name.
%
% tec2mat's own output (info, zone_1, zone_2, ...) addresses variables by
% position (var_1, var_2, ...) because a real Tecplot variable name is
% frequently not a valid MATLAB identifier (e.g. "X [ft]"). When the
% variable names in a particular file ARE known in advance to be valid
% identifiers, this function lets you address them by name instead, e.g.
% plot(zones(1).x, zones(1).pressure) instead of plot(zone_1.var_1,
% zone_1.var_3).
%
% Call:
%     zones = tecmat2struct(d)
%
% Args:
%     d: struct returned by load() on a tec2mat .mat file, i.e. the struct
%        has fields info, zone_1, zone_2, ... as documented in tec2mat.
%
% Returns:
%     zones: 1 x info.num_zones struct array. Every element has the same
%            fields, so this is a normal MATLAB struct array (zones(i),
%            [zones.pressure], etc. all work):
%       .title, .zone_type, .solution_time, .strand_id
%       .I, .J, .K                     ordered zones; [] for FE zones
%       .num_nodes, .num_elements      FE zones; [] for ordered zones
%       .node_map                      FE zones (resolved through
%                                       node_map_shared_from if the zone
%                                       shares connectivity); [] otherwise
%       .face_neighbor_mode, .num_face_connections, .face_connections
%                                       FE zones with face-neighbor data;
%                                       '' / [] / [] otherwise
%       .<variable name>                one field per dataset variable
%                                       whose real name is a valid MATLAB
%                                       identifier (see isvarname). Passive
%                                       in a zone -> []. Shared -> resolved
%                                       to the source zone's data, so every
%                                       zones(i) is self-contained.
%
% Notes:
%  - A variable name that is not a valid MATLAB identifier is skipped with
%    a warning; it remains accessible the original way, via
%    zone_<n>.var_<k> and info.var_names{k}.
%  - Resolving shared variables/connectivity copies the referenced zone's
%    data into every zone that shares it, trading memory for convenience;
%    this can matter for a file whose zones heavily share a static grid
%    across many time steps.
%
% Example:
%     d = load('flow.mat');
%     zones = tecmat2struct(d);
%     plot(zones(1).x, zones(1).pressure);

if ~isfield(d, 'info')
    error('tecmat2struct:badInput', ...
        'Input does not look like a tec2mat output struct (no ''info'' field).');
end

numZones = double(d.info.num_zones);
numVars = double(d.info.num_vars);
varNames = d.info.var_names;

% Reserved field names already used for zone metadata; a variable whose name
% collides with one of these is skipped rather than overwriting metadata.
reserved = {'title', 'zone_type', 'solution_time', 'strand_id', ...
    'I', 'J', 'K', 'num_nodes', 'num_elements', 'node_map', ...
    'face_neighbor_mode', 'num_face_connections', 'face_connections'};

% Validate variable names up front (once) so every zone gets the exact same
% set of variable fields, in the same order -- required for a valid struct
% array, and avoids repeating the same warning once per zone.
validName = false(1, numVars);
for vi = 1:numVars
    name = strtrim(char(varNames{vi}));
    if ~isvarname(name)
        warning('tecmat2struct:invalidName', ...
            ['Variable %d (''%s'') is not a valid MATLAB identifier; skipping. ' ...
             'Access it via zone_<n>.var_%d instead.'], vi, name, vi);
    elseif any(strcmp(name, reserved))
        warning('tecmat2struct:reservedName', ...
            ['Variable %d (''%s'') collides with a zone metadata field; skipping. ' ...
             'Access it via zone_<n>.var_%d instead.'], vi, name, vi);
    else
        validName(vi) = true;
    end
end
varIdx = find(validName);

for zi = numZones:-1:1  % descending so the first iteration preallocates zones(1)
    zsrc = d.(sprintf('zone_%d', zi));

    out = struct( ...
        'title', zsrc.title, ...
        'zone_type', zsrc.zone_type, ...
        'solution_time', zsrc.solution_time, ...
        'strand_id', zsrc.strand_id, ...
        'I', [], 'J', [], 'K', [], ...
        'num_nodes', [], 'num_elements', [], 'node_map', [], ...
        'face_neighbor_mode', '', 'num_face_connections', [], ...
        'face_connections', []);

    if isfield(zsrc, 'I')
        out.I = zsrc.I;
        out.J = zsrc.J;
        out.K = zsrc.K;
    end
    if isfield(zsrc, 'num_nodes')
        out.num_nodes = zsrc.num_nodes;
        out.num_elements = zsrc.num_elements;
    end
    if isfield(zsrc, 'node_map')
        out.node_map = zsrc.node_map;
    elseif isfield(zsrc, 'node_map_shared_from')
        srcZone = d.(sprintf('zone_%d', zsrc.node_map_shared_from));
        out.node_map = srcZone.node_map;
    end
    if isfield(zsrc, 'face_neighbor_mode')
        out.face_neighbor_mode = zsrc.face_neighbor_mode;
        out.num_face_connections = zsrc.num_face_connections;
        out.face_connections = zsrc.face_connections;
    end

    for vi = varIdx
        name = strtrim(char(varNames{vi}));
        status = zsrc.var_status{vi};
        if strcmp(status, 'shared')
            srcIdx = zsrc.var_shared_from(vi);
            srcZone = d.(sprintf('zone_%d', srcIdx));
            out.(name) = srcZone.(sprintf('var_%d', vi));
        else
            % 'active' carries real data; 'passive' is already [].
            out.(name) = zsrc.(sprintf('var_%d', vi));
        end
    end

    zones(zi) = out;
end
end
