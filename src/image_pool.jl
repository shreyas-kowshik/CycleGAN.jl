mutable struct ImagePool
    pool_size
    pool
end

ImagePool(pool_size::Int) = ImagePool(pool_size,[])

function query(ip::ImagePool,images)
    if ip.pool_size > length(ip.pool)
        push!(ip.pool,images) 
        return ip.pool
    else
        pool_copy = ip.pool
        ip.pool[1:end-1] = pool_copy[2:end]
        ip.pool[end] = images
        
        p = rand()
        if p > 0.5
           return images 
        else
            return ip.pool[1]
        end
    end
end