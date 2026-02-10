#ifndef DATASOURCE_HPP
#define DATASOURCE_HPP

#include "../isax/iSAXTypes.hpp"

#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstring>

namespace daisy
{

    using idx_t = unsigned long long;

    class DataSource
    {
    public:
        virtual ~DataSource() = default;

        virtual bool nextRecord(float *record_data) = 0;

        virtual bool hasNext() const = 0;

        virtual idx_t getDim() const = 0;

        virtual idx_t getTotalRecords() const = 0;

        virtual void reset() = 0;

        virtual idx_t getCurrentPosition() const = 0;
    };

    class InMemoryDataSource : public DataSource
    {
    private:
        const float *database;
        idx_t n_database;
        idx_t dim;
        idx_t current_pos;

    public:
        InMemoryDataSource(const float *database, idx_t n_database, idx_t dim)
            : database(database), n_database(n_database), dim(dim), current_pos(0)
        {
        }

        bool nextRecord(float *record_data) override
        {
            if (current_pos >= n_database)
                return false;

            std::copy(database + current_pos * dim,
                      database + current_pos * dim + dim,
                      record_data);

            current_pos++;
            return true;
        }

        bool hasNext() const override
        {
            return current_pos < n_database;
        }

        idx_t getDim() const override
        {
            return dim;
        }

        idx_t getTotalRecords() const override
        {
            return n_database;
        }

        void reset() override
        {
            current_pos = 0;
        }

        idx_t getCurrentPosition() const override
        {
            return current_pos;
        }
    };

    class FileDataSource : public DataSource
    {
    private:
        const char *filename;
        idx_t n_database;
        idx_t dim;
        idx_t current_pos;
        FILE *file;
        bool file_owned;

        void determineTotalRecords()
        {
            if (file == nullptr || dim == 0)
                return;

            long current_pos = ftell(file);
            fseek(file, 0, SEEK_END);
            long file_size = ftell(file);
            fseek(file, current_pos, SEEK_SET);

            if (file_size > 0 && dim > 0)
            {
                n_database = file_size / (sizeof(float) * dim);
            }
        }

    public:
        FileDataSource(const char *filename, idx_t dim, idx_t n_database = 0)
            : filename(filename), n_database(n_database), dim(dim), current_pos(0),
              file(nullptr), file_owned(true)
        {
            file = fopen(filename, "rb");
            if (file == nullptr)
            {
                fprintf(stderr, "Error: Could not open file %s\n", filename);
                throw std::runtime_error("Failed to open file: " + std::string(filename));
            }

            if (n_database == 0)
            {
                determineTotalRecords();
            }
        }

        FileDataSource(FILE *file_opened, idx_t dim, idx_t n_database = 0, bool file_owned = false)
            : filename(nullptr), n_database(n_database), dim(dim), current_pos(0),
              file(file_opened), file_owned(file_owned)
        {
            if (file == nullptr)
            {
                throw std::runtime_error("FileDataSource: file handle is null");
            }

            if (n_database == 0)
            {
                determineTotalRecords();
            }
        }

        ~FileDataSource()
        {
            if (file_owned && file != nullptr)
            {
                fclose(file);
                file = nullptr;
            }
        }

        FileDataSource(const FileDataSource &) = delete;
        FileDataSource &operator=(const FileDataSource &) = delete;

        FileDataSource(FileDataSource &&other) noexcept
            : filename(other.filename), n_database(other.n_database), dim(other.dim),
              current_pos(other.current_pos), file(other.file), file_owned(other.file_owned)
        {
            other.file = nullptr;
            other.file_owned = false;
        }

        bool nextRecord(float *record_data) override
        {
            if (file == nullptr || !hasNext())
                return false;

            size_t items_read = fread(record_data, sizeof(float), dim, file);
            if (items_read != dim)
            {
                return false;
            }

            current_pos++;
            return true;
        }

        bool hasNext() const override
        {
            if (file == nullptr)
                return false;

            if (n_database > 0)
            {
                return current_pos < n_database;
            }
            else
            {

                FILE *mutable_file = const_cast<FILE *>(file);
                long current_pos_saved = ftell(mutable_file);
                if (current_pos_saved < 0)
                    return false;

                float dummy;
                size_t items_read = fread(&dummy, sizeof(float), 1, mutable_file);
                if (items_read == 0)
                {
                    return false;
                }

                fseek(mutable_file, current_pos_saved, SEEK_SET);
                return true;
            }
        }

        idx_t getDim() const override
        {
            return dim;
        }

        idx_t getTotalRecords() const override
        {
            return n_database;
        }

        void reset() override
        {
            if (file != nullptr)
            {
                rewind(file);
                current_pos = 0;
            }
        }

        idx_t getCurrentPosition() const override
        {
            return current_pos;
        }

        FILE *getFileHandle() const
        {
            return file;
        }

        const char *getFilename() const
        {
            return filename;
        }
    };

}

#endif
