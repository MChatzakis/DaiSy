#include "TimerManager.hpp"

namespace daisy
{

    static char *timer_names[] = {
        (char *)"INPUT",
        (char *)"BUFFER",
        (char *)"INDEX",
        (char *)"TOTAL_INDEX_BUFFER",
        (char *)"QA",
        (char *)"WS",
        (char *)"PQ_FILL",
        (char *)"PQ_PREPRO",
        (char *)"PQ_PROCESS",
        (char *)"COMM",
        (char *)"COLLECT",
        (char *)"TOTAL",
        (char *)"QA_PREPRO"};

    static int timer_name_count()
    {
        return static_cast<int>(sizeof(timer_names) / sizeof(char *));
    }

    void timer_init(timer_manager_t *timer_manager)
    {
        if (!timer_manager->timer_map)
        {
            timer_manager->timer_map = new std::unordered_map<std::string, timer_data_t>();
        }
        else
        {
            timer_manager->timer_map->clear();
        }

        for (int i = 0; i < timer_name_count(); i++)
        {
            timer_add(timer_manager, timer_names[i]);
        }
    }

    void timer_destroy(timer_manager_t *timer_manager)
    {
        for (int i = 0; i < timer_name_count(); i++)
        {
            timer_remove(timer_manager, timer_names[i]);
        }

        delete timer_manager->timer_map;
        timer_manager->timer_map = nullptr;
    }

    void timer_add(timer_manager_t *timer_manager, const char *timer_name)
    {
        auto &map = *timer_manager->timer_map;
        if (map.find(timer_name) != map.end())
        {
            return;
        }

        timer_data_t timer{};
        timer.time = 0;
        map.emplace(timer_name, timer);
    }

    void timer_remove(timer_manager_t *timer_manager, const char *timer_name)
    {
        timer_manager->timer_map->erase(timer_name);
    }

    void timer_start(timer_manager_t *timer_manager, const char *timer_name)
    {
        auto it = timer_manager->timer_map->find(timer_name);
        if (it == timer_manager->timer_map->end())
        {
            std::printf("Start: Timer %s does not exist. Ignoring\n", timer_name);
            return;
        }

        gettimeofday(&it->second.start, nullptr);
    }

    void timer_start_force(timer_manager_t *timer_manager, const char *timer_name)
    {
        auto it = timer_manager->timer_map->find(timer_name);
        if (it == timer_manager->timer_map->end())
        {
            timer_add(timer_manager, timer_name);
            it = timer_manager->timer_map->find(timer_name);
        }

        gettimeofday(&it->second.start, nullptr);
    }

    void timer_stop(timer_manager_t *timer_manager, const char *timer_name)
    {
        auto it = timer_manager->timer_map->find(timer_name);
        if (it == timer_manager->timer_map->end())
        {
            std::printf("Stop: Timer %s does not exist. Ignoring\n", timer_name);
            return;
        }

        timeval end{};
        gettimeofday(&end, nullptr);

        double tS = it->second.start.tv_sec * 1000000.0 + it->second.start.tv_usec;
        double tE = end.tv_sec * 1000000.0 + end.tv_usec;
        it->second.time += (tE - tS);
    }

    double timer_get(timer_manager_t *timer_manager, const char *timer_name)
    {
        auto it = timer_manager->timer_map->find(timer_name);
        if (it == timer_manager->timer_map->end())
        {
            std::printf("Get: Timer %s does not exist. Ignoring\n", timer_name);
            return -1;
        }
        return it->second.time;
    }

    void timer_set(timer_manager_t *timer_manager, const char *timer_name, const double val)
    {
        (void)val;
        auto it = timer_manager->timer_map->find(timer_name);
        if (it == timer_manager->timer_map->end())
        {
            std::printf("Set: Timer %s does not exist. Ignoring\n", timer_name);
            return;
        }
        it->second.time = 0;
    }

    void timer_log(timer_manager_t *timer_manager, FILE *stream)
    {
        std::fprintf(stream, "===== TIMERS =====\n");
        for (int i = 0; i < timer_name_count(); i++)
        {
            double res = timer_get(timer_manager, timer_names[i]) / 1000000.0;
            std::fprintf(stream, "%s: %fs", timer_names[i], res);
            if (i > 0 && i % 5 == 0)
            {
                std::fprintf(stream, "\n");
            }
            else if (i != timer_name_count() - 1)
            {
                std::fprintf(stream, " | ");
            }
        }
        std::fprintf(stream, "\n==================\n");
    }

    void timer_to_csv_file(timer_manager_t *timer_manager, const char *filename, bool include_header, bool append)
    {
        const char *mode = append ? "a" : "w";
        FILE *stream = std::fopen(filename, mode);
        if (!stream)
        {
            std::printf("Could not open file %s\n", filename);
            std::exit(EXIT_FAILURE);
        }

        if (include_header)
        {
            for (int i = 0; i < timer_name_count(); i++)
            {
                std::fprintf(stream, "%s", timer_names[i]);
                if (i != timer_name_count() - 1)
                {
                    std::fprintf(stream, ",");
                }
            }
            std::fprintf(stream, "\n");
        }

        for (int i = 0; i < timer_name_count(); i++)
        {
            double res = timer_get(timer_manager, timer_names[i]) / 1000000.0;
            std::fprintf(stream, "%f", res);
            if (i != timer_name_count() - 1)
            {
                std::fprintf(stream, ",");
            }
        }

        std::fclose(stream);
    }

    double *timer_to_array(timer_manager_t *timer_manager, int *size)
    {
        double *res = static_cast<double *>(std::malloc(sizeof(double) * timer_name_count()));
        if (!res)
        {
            std::printf("Could not allocate memory for timer array\n");
            std::exit(EXIT_FAILURE);
        }

        for (int i = 0; i < timer_name_count(); i++)
        {
            res[i] = timer_get(timer_manager, timer_names[i]) / 1000000.0;
        }

        if (size)
        {
            *size = timer_name_count();
        }

        return res;
    }

    char **timer_get_names(timer_manager_t *timer_manager)
    {
        (void)timer_manager;
        return timer_names;
    }

    void timer_print(timer_manager_t *timer_manager, const char *timer_name, FILE *stream)
    {
        double res = timer_get(timer_manager, timer_name) / 1000000.0;
        std::fprintf(stream, "%s: %fs\n", timer_name, res);
    }

} // namespace daisy
