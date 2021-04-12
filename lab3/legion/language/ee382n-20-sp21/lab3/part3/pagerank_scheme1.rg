import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c

fspace Page {
  rank         : double;
  next_rank    : double;
  l_p          : int32;
  delta        : double;
}

fspace Link(r : region(Page)) {
  src     : ptr(Page, r);
  dest    : ptr(Page, r);
}

terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
                      r_links   : region(Link(r_pages)),
                      damp      : double,
                      num_pages : uint64,
                      filename  : int8[512])
where
  reads writes(r_pages, r_links)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for page in r_pages do 
    page.rank = 1.0 / num_pages
    page.l_p = 0  
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = dynamic_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = dynamic_cast(ptr(Page, r_pages), page_ids[1])

    link.src = src_page
    link.dest = dst_page
    -- Add +1 outgoing page
    src_page.l_p += 1
  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end

--
-- TODO: Implement PageRank. You can use as many tasks as you want.
--
-- Initialize all next ranks to 1 - d/n
task initialize_iter(r_pages   : region(Page),
                     damp      : double,
                     num_pages : uint64)
where
  reads (r_pages.rank, r_pages.l_p),
  writes (r_pages.next_rank, r_pages.delta)
do
  for page in r_pages do
    -- Note another way to do this is to initialize this to (1 - damp) / num_pages and forego final_page_rank_iter
    -- But in doing so would incur many more multiplications... which one is more worthy?
    -- Multiplication cycles or an extra sync?
    page.next_rank = ((1 - damp) / num_pages)
    page.delta = page.rank / page.l_p
  end
end

task page_rank_iter(r_pages_main : region(Page),
                    r_pages  : region(Page),
                    r_links  : region(Link(r_pages_main)),
                    damp      : double)
where
  reads writes(r_pages_main.next_rank),
  reads (r_links, r_pages_main.rank, r_pages_main.l_p, r_pages_main.delta)
do
  for link in r_links do  -- Can this be improved?
    link.dest.next_rank += damp * (link.src.delta)
    -- c.printf("Curr page %d Next page %d SRC Rank = %f SRC Lp = %f, Final=%f\n", link.dest, link.src, link.src.rank, link.src.l_p, link.dest.next_rank)
  end
end

task check_converged(r_pages     : region(Page),
                     error_check : double)
where
  reads (r_pages.rank, r_pages.next_rank),
  writes (r_pages.rank)
do
  var converged = true 
  var l2_norm = 0.0

  for page in r_pages do
    l2_norm += (page.rank - page.next_rank) * (page.rank - page.next_rank)
    -- c.printf("Page %d Rank %f Next Rank %f\n", page, page.rank, page.next_rank)
    page.rank = page.next_rank
  end
  return l2_norm <= error_check * error_check 
end

task final_page_rank_iter(r_pages  : region(Page),
                          damp      : double,
                          num_pages : uint64)
where
  reads writes  (r_pages.next_rank)
do
  for page in r_pages do
    page.next_rank = (page.next_rank * damp) + ((1 - damp)/ num_pages)
  end
end

task norm_partition(r_pages : region(Page))
where
  reads(r_pages.rank, r_pages.next_rank),
  writes(r_pages.rank)
do
  var l2_norm = 0.0
  for page in r_pages do
    var change = (page.next_rank - page.rank)
    l2_norm += change*change
    page.rank = page.next_rank
  end
  return l2_norm
end

task dump_ranks(r_pages  : region(Page),
                filename : int8[512])
where
  reads(r_pages.rank)
do
  var f = c.fopen(filename, "w")
  for page in r_pages do c.fprintf(f, "%g\n", page.rank) end
  c.fclose(f)
end

fspace norm
{
  value: double
}

task toplevel()
  var config : PageRankConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Pages  : %11lu *\n",  config.num_pages)
  c.printf("* Number of Links  : %11lu *\n",  config.num_links)
  c.printf("* Damping Factor   : %11.4f *\n", config.damp)
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("* # Parallel Tasks : %11u *\n",   config.parallelism)
  c.printf("**********************************\n")

  -- Create a region of pages
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  var r_links = region(ispace(ptr, config.num_links), Link(r_pages))

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)

  var c_pages = ispace(int1d, config.parallelism)
  var c_links = ispace(int1d, config.parallelism)
  var p_links = partition(equal, r_links, c_links)
  var p_pages = image(r_pages, p_links, r_links.dest)
  var p_pages_equal = partition(equal, r_pages, c_pages)

  var num_iterations = 0
  var converged = false
  var r_norm = region(ispace(int1d, config.parallelism),norm)
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1

    for c in p_pages_equal.colors do
      initialize_iter(p_pages_equal[c], config.damp, config.num_pages)
    end

    for c in p_links.colors do
      page_rank_iter(r_pages, p_pages[c], p_links[c], config.damp)
    end

    for c in p_pages_equal.colors do
      r_norm[c].value = norm_partition(p_pages_equal[c])--,config.error_bound)
    end

    var l2_norm = 0.0
    for x in r_norm do
      l2_norm += x.value
    end    
     
    converged = l2_norm <= config.error_bound*config.error_bound
    
    if(num_iterations >= config.max_iterations) then
      break
    end
  end
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)
