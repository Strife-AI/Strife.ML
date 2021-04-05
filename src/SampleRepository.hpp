#pragma once

namespace StrifeML
{
    template<typename TSample>
    class SampleSet;

    template<typename TSample>
    struct IGroupedSampleView
    {
        virtual ~IGroupedSampleView() = default;

        virtual void AddSample(const TSample& sample, int sampleId) = 0;
    };

    template<typename TSample, typename TSelector>
    class GroupedSampleView : public IGroupedSampleView<TSample>
    {
    public:
        GroupedSampleView(SampleSet<TSample>* owner)
            : _owner(owner)
        {

        }

        GroupedSampleView* GroupBy(std::function<TSelector(const TSample& sample)> selector)
        {
            _selector = selector;
            return this;
        }

        bool TryPickRandomSequence(gsl::span <TSample> outSamples);

        void AddSample(const TSample& sample, int sampleId) override
        {
            if (_selector == nullptr)
            {
                return;
            }

            _samplesBySelectorType[_selector(sample)].push_back(sampleId);
        }

    private:
        SampleSet<TSample>* _owner;
        std::function<TSelector(const TSample& sample)> _selector;
        std::unordered_map <TSelector, std::vector<int>> _samplesBySelectorType;
        std::vector<const std::vector<int>*> _validSampleGroups;
    };

    template<typename TSample>
    class SampleSet
    {
    public:
        SampleSet(RandomNumberGenerator& rng)
            : _rng(rng)
        {

        }

        bool TryGetSampleById(int sampleId, TSample& outSample)
        {
            if (sampleId < 0 || sampleId >= _serializedSamples.size())
            {
                return false;
            }

            auto& serializedSample = _serializedSamples[sampleId];
            ObjectSerializer serializer(serializedSample.bytes, true);
            outSample.input.Serialize(serializer);
            outSample.output.Serialize(serializer);

            return !serializer.hadError;
        }

        int AddSample(const TSample& sample)
        {
            _serializedSamples.emplace_back();
            int sampleId = _serializedSamples.size() - 1;
            auto& serializedObject = _serializedSamples[sampleId];
            ObjectSerializer serializer(serializedObject.bytes, false);

            // This is safe because the serializer is in reading mode
            auto& mutableSample = const_cast<TSample&>(sample);

            mutableSample.input.Serialize(serializer);
            mutableSample.output.Serialize(serializer);

            for (auto& group : _groupedSamplesViews)
            {
                group->AddSample(sample, sampleId);
            }

            return sampleId;
        }

        template<typename TSelector>
        GroupedSampleView<TSample, TSelector>* CreateGroupedView()
        {
            auto group = std::make_unique<GroupedSampleView<TSample, TSelector>>(this);
            auto ptr = group.get();
            _groupedSamplesViews.emplace_back(std::move(group));
            return ptr;
        }

        RandomNumberGenerator& GetRandomNumberGenerator() const
        {
            return _rng;
        }

    private:
        std::vector <SerializedObject> _serializedSamples;
        std::vector <std::unique_ptr<IGroupedSampleView<TSample>>> _groupedSamplesViews;
        RandomNumberGenerator& _rng;
    };

    template<typename TSample, typename TSelector>
    bool GroupedSampleView<TSample, TSelector>::TryPickRandomSequence(gsl::span <TSample> outSamples)
    {
        int minSampleId = outSamples.size() - 1;
        _validSampleGroups.clear();

        for (const auto& groupPair : _samplesBySelectorType)
        {
            const auto& group = groupPair.second;
            if (group.empty())
            {
                continue;
            }

            // The largest sample id in this group isn't big enough to have N samples before it
            if (group[group.size() - 1] < minSampleId)
            {
                continue;
            }

            _validSampleGroups.push_back(&group);
        }

        if (_validSampleGroups.empty())
        {
            return false;
        }

        auto& rng = _owner->GetRandomNumberGenerator();
        int randomSelector = rng.RandInt(0, _validSampleGroups.size() - 1);
        auto& groupToSampleFrom = *_validSampleGroups[randomSelector];
        int groupIndexStart = 0;
        int endSampleId = 0;
        bool found = false;

        while (groupIndexStart < groupToSampleFrom.size())
        {
            int groupIndex = rng.RandInt(groupIndexStart, groupToSampleFrom.size() - 1);
            if (groupToSampleFrom[groupIndex] < minSampleId)
            {
                groupIndexStart = groupIndex + 1;
            }
            else
            {
                endSampleId = groupIndex;
                found = true;
                break;
            }
        }

        // Should be impossible because we checked to make sure the list had at least one sample id big enough
        assert(found);

        for (int i = 0; i < outSamples.size(); ++i)
        {
            int sampleId = endSampleId - (outSamples.size() - 1 - i);
            bool gotSample = _owner->TryGetSampleById(sampleId, outSamples[i]);
            assert(gotSample);
        }

        return true;
    }

    template<typename TSample>
    class SampleRepository
    {
    public:
        SampleRepository(RandomNumberGenerator& rng)
            : _rng(rng)
        {

        }

        SampleSet<TSample>* CreateSampleSet(const char* name)
        {
            // TODO check for duplicate
            _sequencesByName[name] = std::make_unique<SampleSet<TSample>>(_rng);
            return _sequencesByName[name].get();
        }

    private:
        std::unordered_map <std::string, std::unique_ptr<SampleSet<TSample>>> _sequencesByName;
        RandomNumberGenerator& _rng;
    };
}