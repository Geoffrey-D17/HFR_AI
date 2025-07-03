function jday = julian(date)
    % Convert a datetime or datenum to Julian Day
    if isdatetime(date)
        date = datenum(date);
    end
    jday = date + 1721058.5; % MATLAB datenum starts at year 0, Julian day at 4713 BC
end